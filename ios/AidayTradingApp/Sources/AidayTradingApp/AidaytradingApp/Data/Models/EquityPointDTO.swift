import Foundation

struct EquityPointDTO: Decodable {
    let timestamp: Date
    let equity: Decimal

    private enum CodingKeys: String, CodingKey {
        case timestamp = "ts"
        case equity
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rawTimestamp = try container.decode(Double.self, forKey: .timestamp)
        timestamp = Date(timeIntervalSince1970: rawTimestamp / 1000)
        equity = try container.decode(Decimal.self, forKey: .equity)
    }

    func toEntity() -> EquityPoint {
        EquityPoint(timestamp: timestamp, equity: equity)
    }
}
