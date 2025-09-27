import Foundation

protocol PushNotificationRegistering {
    func register(deviceToken: String, accessToken: String) async throws
}

struct PushNotificationService: PushNotificationRegistering {
    private let apiClient: APIClientProtocol

    init(apiClient: APIClientProtocol = APIClient()) {
        self.apiClient = apiClient
    }

    func register(deviceToken: String, accessToken: String) async throws {
        let timezone = TimeZone.current.identifier
        try await apiClient.send(NotificationRequest.register(deviceToken: deviceToken, platform: "ios", timezone: timezone, accessToken: accessToken))
    }
}
