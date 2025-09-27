import Foundation

enum NotificationRequest: APIRequestConvertible {
    case register(deviceToken: String, platform: String, timezone: String, accessToken: String)

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
            }
        }
    }
}
