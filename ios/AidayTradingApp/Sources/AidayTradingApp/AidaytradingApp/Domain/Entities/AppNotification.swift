import Foundation

struct AppNotification: Identifiable, Equatable, Hashable, Codable {
    enum Kind: String, Codable, CaseIterable {
        case report
        case botEvent
        case system
    }

    enum Origin: String, Codable {
        case push
        case realtime
    }

    let id: String
    let title: String
    let body: String
    let timestamp: Date
    let kind: Kind
    let origin: Origin
    let payload: Data

    var payloadDictionary: [String: Any]? {
        (try? JSONSerialization.jsonObject(with: payload) as? [String: Any])
    }

    func formattedTimestamp(for timeZone: TimeZone) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        formatter.timeZone = timeZone
        return formatter.string(from: timestamp)
    }

    func centralTimestampString() -> String {
        formattedTimestamp(for: AppConfig.centralTimeZone)
    }
}
