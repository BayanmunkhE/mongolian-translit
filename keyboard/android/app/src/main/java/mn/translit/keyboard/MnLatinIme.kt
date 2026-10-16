package mn.translit.keyboard

import android.graphics.Color
import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.widget.Button
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * Pinyin-style keyboard: type Latin, the strip shows Cyrillic candidates, space commits the first one,
 * tapping a candidate commits that one (and remembers it when it was not the first).
 * Words with no Latin letters (numbers, emoji) and the raw Latin itself are always offered too.
 */
class MnLatinIme : InputMethodService() {
    private lateinit var dict: WordDictionary
    private var model: Transliterator? = null
    private val worker = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var strip: LinearLayout
    private var composing = StringBuilder()
    private var shift = false
    private var request = 0
    /** The last two committed Latin words: context for the model (cleared by punctuation and Enter). */
    private val recentLatin = ArrayDeque<String>()

    override fun onCreate() {
        super.onCreate()
        dict = WordDictionary(this)
        worker.execute { model = Transliterator(this) }  // ~1 s to load both graphs
    }

    override fun onCreateInputView(): View {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.parseColor("#ECE8DF")) }
        strip = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; minimumHeight = dp(44) }
        root.addView(HorizontalScrollView(this).apply { addView(strip) })
        for (row in listOf("qwertyuiop", "asdfghjkl", "^zxcvbnm<", " ,.!?")) root.addView(keyRow(row))
        showSuggestions(emptyList())
        return root
    }

    private fun keyRow(keys: String): LinearLayout {
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for (c in keys) {
            val b = Button(this).apply {
                text = when (c) { '^' -> "⇧"; '<' -> "⌫"; ' ' -> "зай"; else -> c.toString() }
                textSize = 18f
                setOnClickListener { onKey(c) }
                layoutParams = LinearLayout.LayoutParams(0, dp(48), if (c == ' ') 5f else 1f)
            }
            row.addView(b)
        }
        return row
    }

    private fun onKey(c: Char) {
        when (c) {
            '^' -> shift = !shift
            '<' -> if (composing.isNotEmpty()) { composing.setLength(composing.length - 1); updateComposing() }
                   else currentInputConnection?.deleteSurroundingText(1, 0)
            ' ' -> commitFirst(" ")
            ',', '.', '!', '?' -> commitFirst(c.toString())
            else -> { composing.append(if (shift) c.uppercaseChar() else c); shift = false; updateComposing() }
        }
    }

    private fun updateComposing() {
        currentInputConnection?.setComposingText(composing, 1)
        val latin = composing.toString()
        if (latin.isEmpty()) { showSuggestions(emptyList()); return }
        val id = ++request
        val prefix = recentLatin.toList()
        val fromDict = dict.lookup(latin)
        // a user pick or a dictionary hit answers at once; the model refines it with context when there is any
        if (fromDict != null) showSuggestions(fromDict.words + latin) else showSuggestions(listOf(latin))
        if (fromDict != null && prefix.isEmpty()) return
        worker.execute {
            val cyr = model?.convertInContext(prefix, latin) ?: return@execute
            ui.post {
                if (id != request) return@post
                val rest = (fromDict?.words ?: emptyList()).filter { !it.equals(cyr, ignoreCase = true) }
                showSuggestions(listOf(cyr) + rest + latin)
            }
        }
    }

    private var suggestions: List<String> = emptyList()

    /** The model capitalises a lone word like a sentence start; mid-sentence the user's casing wins. */
    private fun matchCase(word: String): String {
        val latin = composing.toString()
        return if (latin.isNotEmpty() && latin[0].isLowerCase() && word.isNotEmpty() && word[0].isUpperCase())
            word[0].lowercaseChar() + word.substring(1) else word
    }

    private fun showSuggestions(words: List<String>) {
        suggestions = words.map { matchCase(it) }.distinct()
        strip.removeAllViews()
        for ((i, w) in suggestions.withIndex()) {
            strip.addView(TextView(this).apply {
                text = w; textSize = 18f; gravity = Gravity.CENTER
                setPadding(dp(14), dp(10), dp(14), dp(10))
                setTextColor(Color.parseColor(if (i == 0) "#2563A8" else "#16213A"))
                setOnClickListener { commit(w, " ", i) }
            })
        }
    }

    private fun commitFirst(suffix: String) {
        if (composing.isEmpty()) { currentInputConnection?.commitText(suffix, 1); return }
        commit(suggestions.firstOrNull() ?: composing.toString(), suffix, 0)
    }

    private fun commit(word: String, suffix: String, index: Int) {
        val latin = composing.toString()
        if (index > 0 && latin.isNotEmpty()) dict.remember(latin, word)
        currentInputConnection?.commitText(word + suffix, 1)
        if (latin.isNotEmpty()) { recentLatin.addLast(latin); while (recentLatin.size > 2) recentLatin.removeFirst() }
        if (suffix.trim().isNotEmpty()) recentLatin.clear()  // punctuation ends the context window
        composing.setLength(0)
        showSuggestions(emptyList())
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_BACK && composing.isNotEmpty()) { commitFirst(""); return true }
        return super.onKeyDown(keyCode, event)
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
}
