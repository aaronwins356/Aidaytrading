import Foundation
import SwiftUI
import UIKit
import UserNotifications

@MainActor
final class NotificationController: ObservableObject {
    @Published private(set) var pendingTab: MainTabView.Tab?

    private let pushService: PushNotificationRegistering
    private let localScheduler: LocalNotificationScheduling
    private var currentSession: UserSessionContext?
    private var cachedFCMToken: String?
    private var hasRequestedAuthorization = false
    private var lastRegisteredToken: String?

    init(
        pushService: PushNotificationRegistering = PushNotificationService(),
        localScheduler: LocalNotificationScheduling = LocalNotificationScheduler()
    ) {
        self.pushService = pushService
        self.localScheduler = localScheduler
    }

    func updateSessionState(_ state: SessionStore.SessionState) {
        switch state {
        case .authenticated(let context):
            currentSession = context
            requestNotificationAuthorizationIfNeeded()
            Task {
                await registerIfPossible()
            }
        default:
            currentSession = nil
            cachedFCMToken = nil
            lastRegisteredToken = nil
        }
    }

    func registerFCMToken(_ token: String) {
        cachedFCMToken = token
        Task {
            await registerIfPossible()
        }
    }

    func handleRemoteNotification(userInfo: [AnyHashable: Any]) {
        guard let target = userInfo["target"] as? String else { return }
        switch target.lowercased() {
        case "trades":
            pendingTab = .trades
        case "calendar":
            pendingTab = .calendar
        case "admin":
            pendingTab = .admin
        default:
            pendingTab = .home
        }
    }

    func consumePendingTab() -> MainTabView.Tab? {
        defer { pendingTab = nil }
        return pendingTab
    }

    func scheduleRealtimeStallNotification() {
        Task {
            await localScheduler.scheduleRealtimeStallNotification()
        }
    }

    func cancelRealtimeStallNotification() {
        localScheduler.cancelRealtimeStallNotification()
    }

    private func requestNotificationAuthorizationIfNeeded() {
        guard !hasRequestedAuthorization else { return }
        hasRequestedAuthorization = true
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, _ in
            if granted {
                DispatchQueue.main.async {
                    UIApplication.shared.registerForRemoteNotifications()
                }
            }
        }
    }

    private func registerIfPossible() async {
        guard let context = currentSession, let token = cachedFCMToken else { return }
        guard token != lastRegisteredToken else { return }
        do {
            try await pushService.register(deviceToken: token, accessToken: context.tokens.accessToken)
            lastRegisteredToken = token
        } catch {
            // Surface the error via console for diagnostics but avoid alerting end users repeatedly.
            NSLog("Failed to register push token: \(error.localizedDescription)")
        }
    }
}
