import Foundation

enum NotificationRequest: APIRequestConvertible {
    case register(deviceToken: String, platform: String, timezone: String, accessToken: String)
    case preferences(accessToken: String)
    case updatePreferences(NotificationPreferences, accessToken: String)
    case unregister(deviceToken: String, accessToken: String)

    var urlRequest: URLRequest {
        get throws {
            switch self {
            case let .register(deviceToken, platform, timezone, accessToken):
                var request = URLRequest(url: APIEnvironment.baseURL.appending(path: "/api/v1/notifications/devices"))
                request.httpMethod = "POST"
                request.addValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
                request.addValue("application/json", forHTTPHeaderField: "Accept")
                request.addValue("application/json", forHTTPHeaderField: "Content-Type")
                let payload: [String: String] = [
                    "token": deviceToken,
                    "platform": platform,
                    "timezone": timezone
                ]
                request.httpBody = try JSONEncoder().encode(payload)
                return request
            case let .preferences(accessToken):
                var request = URLRequest(url: APIEnvironment.baseURL.appending(path: "/api/v1/notifications/preferences"))
                request.httpMethod = "GET"
                request.addValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
                request.addValue("application/json", forHTTPHeaderField: "Accept")
                return request
            case let .updatePreferences(preferences, accessToken):
                var request = URLRequest(url: APIEnvironment.baseURL.appending(path: "/api/v1/notifications/preferences"))
                request.httpMethod = "PUT"
                request.addValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
                request.addValue("application/json", forHTTPHeaderField: "Accept")
                request.addValue("application/json", forHTTPHeaderField: "Content-Type")
                struct Payload: Encodable {
                    let botEventsEnabled: Bool
                    let reportsEnabled: Bool
                }
                let payload = Payload(botEventsEnabled: preferences.botEventsEnabled, reportsEnabled: preferences.reportsEnabled)
                request.httpBody = try JSONEncoder().encode(payload)
                return request
            case let .unregister(deviceToken, accessToken):
                var request = URLRequest(url: APIEnvironment.baseURL.appending(path: "/api/v1/notifications/devices"))
                request.httpMethod = "DELETE"
                request.addValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
                request.addValue("application/json", forHTTPHeaderField: "Accept")
                let payload = ["token": deviceToken]
                request.httpBody = try JSONEncoder().encode(payload)
                return request
            }
        }
    }
}
