import Foundation

protocol PushNotificationRegistering {
    func register(deviceToken: String, accessToken: String) async throws
}

protocol NotificationPreferencesServicing {
    func fetchPreferences(accessToken: String) async throws -> NotificationPreferences
    func update(preferences: NotificationPreferences, accessToken: String) async throws -> NotificationPreferences
    func unregister(deviceToken: String, accessToken: String) async throws
}

struct PushNotificationService: PushNotificationRegistering, NotificationPreferencesServicing {
    private let apiClient: APIClientProtocol

    init(apiClient: APIClientProtocol = APIClient()) {
        self.apiClient = apiClient
    }

    func register(deviceToken: String, accessToken: String) async throws {
        let timezone = TimeZone.current.identifier
        try await apiClient.send(NotificationRequest.register(deviceToken: deviceToken, platform: "ios", timezone: timezone, accessToken: accessToken))
    }

    func fetchPreferences(accessToken: String) async throws -> NotificationPreferences {
        try await apiClient.send(NotificationRequest.preferences(accessToken: accessToken))
    }

    func update(preferences: NotificationPreferences, accessToken: String) async throws -> NotificationPreferences {
        try await apiClient.send(NotificationRequest.updatePreferences(preferences, accessToken: accessToken))
    }

    func unregister(deviceToken: String, accessToken: String) async throws {
        try await apiClient.send(NotificationRequest.unregister(deviceToken: deviceToken, accessToken: accessToken))
    }
}
