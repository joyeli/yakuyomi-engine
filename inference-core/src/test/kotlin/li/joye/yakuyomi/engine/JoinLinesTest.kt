package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Test

/** 行合併：日文直接相接；英文／韓文行間補空白，行尾連字號斷字接回。 */
class JoinLinesTest {

    @Test fun japanese_lines_join_without_space() {
        assertEquals("その通りじゃ。", joinLines(listOf("その通", "りじゃ。")))
    }

    @Test fun english_lines_join_with_space() {
        assertEquals(
            "ANYWAY, TODAY IS NO DIFFERENT FROM THE MISERABLE PAST,",
            joinLines(listOf("ANYWAY, TODAY IS NO", "DIFFERENT FROM THE", "MISERABLE PAST,")),
        )
    }

    @Test fun hyphenated_word_is_rejoined() {
        assertEquals("NO DIFFERENT", joinLines(listOf("NO DIF-", "FERENT")))
        // 單獨的破折號不是斷字
        assertEquals("WAIT - NO", joinLines(listOf("WAIT -", "NO")))
    }

    @Test fun korean_lines_join_with_space() {
        assertEquals("오늘도 다르지 않다", joinLines(listOf("오늘도", "다르지 않다")))
    }

    @Test fun blank_lines_are_skipped() {
        assertEquals("HELLO THERE", joinLines(listOf(" HELLO ", "", "THERE")))
    }
}
