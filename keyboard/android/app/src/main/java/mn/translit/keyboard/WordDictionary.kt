package mn.translit.keyboard

import android.content.Context
import org.json.JSONObject

/** Candidate list for one Latin word, best first. */
data class Candidates(val words: List<String>, val fromModel: Boolean)

/**
 * Two lookups before the model runs:
 *  - the user's own picks (SharedPreferences): a word the user corrected once comes back corrected;
 *  - the packed dictionary (assets/word_dict.json, built by tools/word_dict.py): the 50k most frequent Latin
 *    words with up to 3 teacher candidates each. Instant, and it is also how English words stay untouched.
 */
class WordDictionary(context: Context) {
    private val prefs = context.getSharedPreferences("user_dict", Context.MODE_PRIVATE)
    private val packed: Map<String, List<String>> by lazy { load(context) }
    /** previous Cyrillic word -> (next word -> count), from tools/word_bigrams.py; ranks the candidates. */
    private val bigrams: Map<String, Map<String, Int>> by lazy { loadBigrams(context) }

    private fun loadBigrams(context: Context): Map<String, Map<String, Int>> = runCatching {
        val root = JSONObject(context.assets.open("bigrams.json").bufferedReader().readText())
        val out = HashMap<String, Map<String, Int>>(root.length() * 2)
        for (prev in root.keys()) {
            val nxt = root.getJSONObject(prev)
            out[prev] = nxt.keys().asSequence().associateWith { nxt.getInt(it) }
        }
        out
    }.getOrDefault(emptyMap())

    private fun load(context: Context): Map<String, List<String>> {
        val text = context.assets.open("word_dict.json").bufferedReader().readText()
        val root = JSONObject(text)
        val out = HashMap<String, List<String>>(root.length() * 2)
        for (key in root.keys()) {
            val cands = root.getJSONObject(key).getJSONArray("candidates")
            out[key] = (0 until cands.length()).map { cands.getJSONArray(it).getString(0) }
        }
        return out
    }

    /**
     * User choice first, then the packed list re-ranked by how often each candidate follows `prevCyrillic`
     * (bigram counts); null when the model has to run.
     */
    fun lookup(latin: String, prevCyrillic: String? = null): Candidates? {
        val key = latin.lowercase()
        val user = prefs.getString(key, null)
        val base = packed[key]
        if (user == null && base == null) return null
        val words = ArrayList<String>()
        if (user != null) words.add(user)
        val follow = prevCyrillic?.let { bigrams[it.lowercase()] }
        val ranked = if (follow == null || base == null) base
                     else base.sortedByDescending { follow[it.lowercase()] ?: 0 }  // stable: ties keep model order
        ranked?.forEach { if (!words.contains(it)) words.add(it) }
        return Candidates(words, fromModel = false)
    }

    /** Remember a pick that was not the first suggestion. */
    fun remember(latin: String, chosen: String) {
        prefs.edit().putString(latin.lowercase(), chosen).apply()
    }
}
