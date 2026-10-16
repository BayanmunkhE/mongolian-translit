package mn.translit.keyboard

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Greedy byte-level decoding with the two ONNX graphs exported by tools/export_onnx.py
 * (encoder.int8.onnx + decoder.int8.onnx, no KV cache: a word is <= ~25 bytes, so re-running the 2-layer
 * decoder over the growing prefix costs ~8 ms per word on a desktop core, ~30-60 ms on a phone).
 * Mirrors OnnxTransliterator in that script: ByT5 token = byte + 3, pad 0, eos 1, decoder start 0.
 */
class Transliterator(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val decoder: OrtSession
    private val eos: Long
    private val start: Long
    private val vocab: Int
    private val cache = object : LinkedHashMap<String, String>(256, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, String>?) = size > 2048
    }

    init {
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
        encoder = env.createSession(context.assets.open("encoder.int8.onnx").readBytes(), opts)
        decoder = env.createSession(context.assets.open("decoder.int8.onnx").readBytes(), opts)
        val meta = JSONObject(context.assets.open("meta.json").bufferedReader().readText())
        eos = meta.getLong("eos_token_id")
        start = meta.getLong("decoder_start_token_id")
        vocab = meta.getInt("vocab_size")
    }

    /**
     * Convert `word` as the last word of "prefix word" (the previous Latin words the user committed) and return
     * only the part after the prefix's own conversion. Context lifts word accuracy from ~75% to ~87% on the test
     * sets (results/keyboard/context_eval.md): у/ү/ө, уу/үү and "bna" → "байна байна" are decided by neighbours.
     */
    @Synchronized
    fun convertInContext(prefix: List<String>, word: String): String {
        if (prefix.isEmpty()) return convert(word)
        val prefixText = prefix.joinToString(" ")
        val full = convert("$prefixText $word")
        val prefixConv = convert(prefixText)
        if (prefixConv.isNotEmpty() && full.lowercase().startsWith(prefixConv.lowercase())) {
            val rest = full.substring(prefixConv.length).trim()
            if (rest.isNotEmpty()) return rest
        }
        return full.split(Regex("\\s+")).lastOrNull() ?: full
    }

    @Synchronized
    fun convert(latin: String, maxNew: Int = 64): String {
        cache[latin]?.let { return it }
        val bytes = latin.toByteArray(Charsets.UTF_8)
        val src = LongArray(bytes.size + 1) { i -> if (i < bytes.size) (bytes[i].toLong() and 0xFF) + 3 else eos }
        val srcShape = longArrayOf(1, src.size.toLong())
        val ids = OnnxTensor.createTensor(env, LongBuffer.wrap(src), srcShape)
        val mask = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(src.size) { 1 }), srcShape)
        val encOut = encoder.run(mapOf("input_ids" to ids, "attention_mask" to mask))
        val hiddenTensor = encOut[0] as OnnxTensor
        val hiddenShape = hiddenTensor.info.shape
        val hidden = OnnxTensor.createTensor(env, hiddenTensor.floatBuffer, hiddenShape)

        val out = ArrayList<Long>().apply { add(start) }
        try {
            for (step in 0 until maxNew) {
                val decIds = OnnxTensor.createTensor(env, LongBuffer.wrap(out.toLongArray()),
                    longArrayOf(1, out.size.toLong()))
                val res = decoder.run(mapOf("decoder_input_ids" to decIds, "encoder_hidden_states" to hidden,
                    "attention_mask" to mask))
                val logits: FloatBuffer = (res[0] as OnnxTensor).floatBuffer
                val offset = (out.size - 1) * vocab
                var best = 0
                var bestVal = Float.NEGATIVE_INFINITY
                for (v in 0 until vocab) {
                    val x = logits.get(offset + v)
                    if (x > bestVal) { bestVal = x; best = v }
                }
                res.close(); decIds.close()
                if (best.toLong() == eos) break
                out.add(best.toLong())
            }
        } finally {
            hidden.close(); encOut.close(); ids.close(); mask.close()
        }
        val outBytes = out.drop(1).filter { it >= 3 }.map { (it - 3).toByte() }.toByteArray()
        val text = String(outBytes, Charsets.UTF_8)
        cache[latin] = text
        return text
    }
}
