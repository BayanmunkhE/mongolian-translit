package mn.translit.keyboard

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONArray
import java.nio.LongBuffer

/**
 * Encoder-only tagger (tools/export_tagger_onnx.py): one forward pass gives a label id per input byte, each label
 * an output string (vocab.json). ~0.5 ms per word on a desktop core, 34 MB int8. Label 1 = <unk>: the byte's
 * character is copied unchanged (the keyboard can fall back to the seq2seq model for such words).
 */
class TaggerTransliterator(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val sess: OrtSession
    private val vocab: List<String>

    init {
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        sess = env.createSession(context.assets.open("tagger.int8.onnx").readBytes(), opts)
        val arr = JSONArray(context.assets.open("tagger_vocab.json").bufferedReader().readText())
        vocab = (0 until arr.length()).map { arr.getString(it) }
    }

    /** Returns the conversion and whether every byte had a known label. */
    @Synchronized
    fun convert(latin: String): Pair<String, Boolean> {
        val bytes = latin.toByteArray(Charsets.UTF_8)
        val ids = LongArray(bytes.size + 1) { i -> if (i < bytes.size) (bytes[i].toLong() and 0xFF) + 3 else 1L }
        val shape = longArrayOf(1, ids.size.toLong())
        val input = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape)
        val mask = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(ids.size) { 1 }), shape)
        val res = sess.run(mapOf("input_ids" to input, "attention_mask" to mask))
        val labels = (res[0].value as Array<LongArray>)[0]
        res.close(); input.close(); mask.close()
        val out = StringBuilder()
        var pos = 0
        var allKnown = true
        for (ch in latin) {
            val n = ch.toString().toByteArray(Charsets.UTF_8).size
            val lab = if (pos < labels.size) vocab[labels[pos].toInt()] else "<unk>"
            if (lab == "<unk>") { out.append(ch); allKnown = false } else out.append(lab)
            pos += n
        }
        return out.toString() to allKnown
    }

    /** Context-aware: convert "prefix word" and keep what follows the prefix's own conversion. */
    @Synchronized
    fun convertInContext(prefix: List<String>, word: String): Pair<String, Boolean> {
        if (prefix.isEmpty()) return convert(word)
        val prefixText = prefix.joinToString(" ")
        val (full, known) = convert("$prefixText $word")
        val (prefixConv, _) = convert(prefixText)
        if (prefixConv.isNotEmpty() && full.lowercase().startsWith(prefixConv.lowercase())) {
            val rest = full.substring(prefixConv.length).trim()
            if (rest.isNotEmpty()) return rest to known
        }
        return (full.split(Regex("\\s+")).lastOrNull() ?: full) to known
    }
}
