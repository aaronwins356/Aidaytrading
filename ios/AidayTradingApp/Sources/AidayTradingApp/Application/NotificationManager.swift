import Combine
import FirebaseMessaging
import Foundation
import SwiftUI
import UIKit
import UserNotifications

@MainActor
final class NotificationManager: NSObject, ObservableObject {
    @Published private(set) var notifications: [AppNotification] = []
    @Published private(set) var pendingTab: MainTabView.Tab?
    @Published private(set) var preferences: NotificationPreferences = .default
    @Published private(set) var canModifyPreferences = false
    @Published private(set) var activeBanner: AppNotification?

    private let pushService: PushNotificationRegistering & NotificationPreferencesServicing
    private let localScheduler: LocalNotificationScheduling
    private let persistence: NotificationPersistenceController
    private var currentSession: UserSessionContext?
    private var cachedFCMToken: String?
    private var lastRegisteredToken: String?
    private var bannerTask: Task<Void, Never>?
    private var realtimeCancellables = Set<AnyCancellable>()
    private var lastRealtimeStatus: Status?
    private var hasRequestedAuthorization = false

    init(
        pushService: PushNotificationRegistering & NotificationPreferencesServicing = PushNotificationService(),
        localScheduler: LocalNotificationScheduling = LocalNotificationScheduler(),
        persistence: NotificationPersistenceController = .shared
    ) {
        self.pushService = pushService
        self.localScheduler = localScheduler
        self.persistence = persistence
        super.init()
    }

    func registerForPushNotifications() {
        guard !hasRequestedAuthorization else { return }
        hasRequestedAuthorization = true
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, _ in
            if granted {
                DispatchQueue.main.async {
                    UIApplication.shared.registerForRemoteNotifications()
                }
            }
        }
    }

    func updateSessionState(_ state: SessionStore.SessionState) {
        switch state {
        case .authenticated(let context):
            currentSession = context
            canModifyPreferences = RoleManager.isAdmin(context.profile)
            registerForPushNotifications()
            Task {
                await loadPreferencesIfNeeded()
                await registerDeviceIfPossible()
            }
            refreshStoredNotifications()
        default:
            currentSession = nil
            cachedFCMToken = nil
            lastRegisteredToken = nil
            canModifyPreferences = false
            preferences = .default
            notifications = []
            hasRequestedAuthorization = false
        }
    }

    func didRegisterForRemoteNotificationsWithDeviceToken(_ deviceToken: Data) {
        Messaging.messaging().apnsToken = deviceToken
    }

    func handleFCMToken(_ token: String) {
        cachedFCMToken = token
        Task { await registerDeviceIfPossible() }
    }

    func handleRemoteNotification(userInfo: [AnyHashable: Any]) {
        guard let notification = parseNotificationPayload(userInfo: userInfo) else { return }
        store(notification: notification)
        if UIApplication.shared.applicationState == .active {
            presentBanner(for: notification)
        }
        pendingTab = tab(for: userInfo)
    }

    func consumePendingTab() -> MainTabView.Tab? {
        defer { pendingTab = nil }
        return pendingTab
    }

    func bind(to client: TradingWebSocketClientProtocol) {
        realtimeCancellables.removeAll()
        client.statusPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in
                guard let self else { return }
                self.handleRealtimeStatus(status)
            }
            .store(in: &realtimeCancellables)

        client.tradePublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] trade in
                guard let self else { return }
                self.handleRealtimeTrade(trade)
            }
            .store(in: &realtimeCancellables)

        client.connectionState
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in
                guard let self else { return }
                switch state {
                case .connected:
                    self.localScheduler.cancelRealtimeStallNotification()
                case .reconnecting:
                    Task { await self.localScheduler.scheduleRealtimeStallNotification() }
                case .disconnected, .connecting:
                    break
                }
            }
            .store(in: &realtimeCancellables)
    }

    func delete(notificationID: String) {
        do {
            try persistence.delete(notificationID: notificationID)
            refreshStoredNotifications()
        } catch {
            NSLog("Failed to delete notification: \(error.localizedDescription)")
        }
    }

    func updatePreferences(_ preferences: NotificationPreferences) async {
        guard let session = currentSession, canModifyPreferences else { return }
        do {
            let updated = try await pushService.update(preferences: preferences, accessToken: session.tokens.accessToken)
            self.preferences = updated
        } catch {
            NSLog("Failed to update notification preferences: \(error.localizedDescription)")
        }
    }

    func unregisterDevice() async {
        guard let session = currentSession, let token = cachedFCMToken else { return }
        do {
            try await pushService.unregister(deviceToken: token, accessToken: session.tokens.accessToken)
            lastRegisteredToken = nil
        } catch {
            NSLog("Failed to unregister device: \(error.localizedDescription)")
        }
    }

    func setBotEvents(enabled: Bool) {
        guard canModifyPreferences else { return }
        var updated = preferences
        updated.botEventsEnabled = enabled
        preferences = updated
        Task { await updatePreferences(updated) }
    }

    func setReports(enabled: Bool) {
        guard canModifyPreferences else { return }
        var updated = preferences
        updated.reportsEnabled = enabled
        preferences = updated
        Task { await updatePreferences(updated) }
    }

    private func loadPreferencesIfNeeded() async {
        guard let session = currentSession else { return }
        do {
            preferences = try await pushService.fetchPreferences(accessToken: session.tokens.accessToken)
        } catch {
            preferences = .default
            NSLog("Failed to load notification preferences: \(error.localizedDescription)")
        }
    }

    private func registerDeviceIfPossible() async {
        guard let session = currentSession, let token = cachedFCMToken else { return }
        guard token != lastRegisteredToken else { return }
        do {
            try await pushService.register(deviceToken: token, accessToken: session.tokens.accessToken)
            lastRegisteredToken = token
        } catch {
            NSLog("Failed to register push token: \(error.localizedDescription)")
        }
    }

    private func parseNotificationPayload(userInfo: [AnyHashable: Any]) -> AppNotification? {
        guard let aps = userInfo["aps"] as? [String: Any] else { return nil }
        let alert = aps["alert"] as? [String: Any]
        let title = (alert?["title"] as? String) ?? "Trading Bot Update"
        let body = (alert?["body"] as? String) ?? ""
        let timestamp = Date()
        var dataPayload: [String: Any] = [:]
        if let data = userInfo["data"] as? [String: Any] {
            dataPayload = data
        } else {
            for (key, value) in userInfo {
                if let key = key as? String, key != "aps" {
                    dataPayload[key] = value
                }
            }
        }
        let type = (dataPayload["type"] as? String) ?? "system"
        let kind: AppNotification.Kind
        switch type.lowercased() {
        case "report":
            kind = .report
        case "bot_event", "bot_status":
            kind = .botEvent
        default:
            kind = .system
        }
        let identifier = (dataPayload["notification_id"] as? String) ?? UUID().uuidString
        let payloadData = (try? JSONSerialization.data(withJSONObject: dataPayload, options: [])) ?? Data()
        return AppNotification(
            id: identifier,
            title: title,
            body: body,
            timestamp: timestamp,
            kind: kind,
            origin: .push,
            payload: payloadData
        )
    }

    private func tab(for userInfo: [AnyHashable: Any]) -> MainTabView.Tab? {
        let target = (userInfo["target"] as? String) ?? ((userInfo["data"] as? [String: Any])?["target"] as? String)
        switch target?.lowercased() {
        case "trades": return .trades
        case "calendar": return .calendar
        case "admin": return .admin
        case "notifications": return .notifications
        case "home": return .home
        default: return nil
        }
    }

    private func store(notification: AppNotification) {
        do {
            try persistence.save(notification: notification)
            refreshStoredNotifications()
        } catch {
            NSLog("Failed to persist notification: \(error.localizedDescription)")
        }
    }

    private func refreshStoredNotifications() {
        do {
            notifications = try persistence.fetchNotifications()
        } catch {
            NSLog("Failed to fetch notifications: \(error.localizedDescription)")
            notifications = []
        }
    }

    private func presentBanner(for notification: AppNotification) {
        activeBanner = notification
        bannerTask?.cancel()
        bannerTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 4 * NSEC_PER_SEC)
            await MainActor.run {
                self?.activeBanner = nil
            }
        }
    }

    private func handleRealtimeStatus(_ status: Status) {
        defer { lastRealtimeStatus = status }
        guard lastRealtimeStatus?.running != status.running else { return }
        let title = status.running ? "Bot Started" : "Bot Stopped"
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        formatter.timeZone = AppConfig.centralTimeZone
        let centralTime = formatter.string(from: Date())
        let body = status.running ? "Automations resumed at \(centralTime) Central." : "Automations paused at \(centralTime) Central."
        let payload: [String: Any] = [
            "type": "bot_event",
            "running": status.running,
            "timestamp": Date().timeIntervalSince1970
        ]
        recordRealtimeNotification(
            id: "bot_status_\(Int(Date().timeIntervalSince1970))",
            title: title,
            body: body,
            kind: .botEvent,
            payload: payload
        )
    }

    private func handleRealtimeTrade(_ trade: Trade) {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "USD"
        formatter.maximumFractionDigits = 2
        let pnlText = formatter.string(from: trade.pnl as NSDecimalNumber) ?? trade.pnl.currencyString()
        let title = "Trade \(trade.side.displayName) \(trade.symbol)"
        let body = "P/L: \(pnlText)"
        let payload: [String: Any] = [
            "type": "trade",
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "pnl": trade.pnl.description
        ]
        recordRealtimeNotification(
            id: "trade_\(trade.id)",
            title: title,
            body: body,
            kind: .system,
            payload: payload
        )
    }

    private func recordRealtimeNotification(id: String, title: String, body: String, kind: AppNotification.Kind, payload: [String: Any]) {
        guard let payloadData = try? JSONSerialization.data(withJSONObject: payload, options: []) else { return }
        let notification = AppNotification(
            id: id,
            title: title,
            body: body,
            timestamp: Date(),
            kind: kind,
            origin: .realtime,
            payload: payloadData
        )
        store(notification: notification)
        presentBanner(for: notification)
    }
}
