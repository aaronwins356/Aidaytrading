import Foundation
import UserNotifications

protocol LocalNotificationScheduling {
    func scheduleRealtimeStallNotification() async
    func cancelRealtimeStallNotification()
}

struct LocalNotificationScheduler: LocalNotificationScheduling {
    private let notificationCenter: UNUserNotificationCenter
    private let identifier = "com.aidaytrading.notifications.realtime-stall"

    init(notificationCenter: UNUserNotificationCenter = .current()) {
        self.notificationCenter = notificationCenter
    }

    func scheduleRealtimeStallNotification() async {
        let settings = await notificationCenter.notificationSettings()
        if settings.authorizationStatus != .authorized {
            _ = try? await notificationCenter.requestAuthorization(options: [.alert, .badge, .sound])
        }

        let content = UNMutableNotificationContent()
        content.title = "Realtime data paused"
        content.body = "We haven't received new trading data recently. Reopen the app to refresh."
        content.sound = .default
        content.userInfo = ["target": "home"]

        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let request = UNNotificationRequest(identifier: identifier, content: content, trigger: trigger)
        try? await notificationCenter.add(request)
    }

    func cancelRealtimeStallNotification() {
        notificationCenter.removePendingNotificationRequests(withIdentifiers: [identifier])
    }
}
