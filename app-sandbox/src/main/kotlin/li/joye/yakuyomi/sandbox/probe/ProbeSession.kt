package li.joye.yakuyomi.sandbox.probe

import androidx.documentfile.provider.DocumentFile

/** 這個行程裡的探針狀態：探針跑在自己的執行緒上，畫面只是看它；轉向、切走都不影響（切走會被記成警告）。 */
object ProbeSession {
    @Volatile
    var running = false

    /** 畫面是否看得見（最上層）。 */
    @Volatile
    var activityVisible = false

    /** 離開過最上層幾次（量到的時間不算數：這台手機切走後只給 5 顆核心）。 */
    @Volatile
    var leftCount = 0

    @Volatile
    var report: ProbeReport? = null
    @Volatile
    var tree: DocumentFile? = null
}
