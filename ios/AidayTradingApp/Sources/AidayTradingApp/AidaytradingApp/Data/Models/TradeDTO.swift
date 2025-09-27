import Foundation

struct TradeDTO: Decodable {
    let id: String
    let symbol: String
    let side: String
    let quantity: Decimal
    let price: Decimal
    let pnl: Decimal
    let pnlPct: Decimal
    let openedAt: Date
    let closedAt: Date?

    private enum CodingKeys: String, CodingKey {
        case id
        case symbol
        case side
        case quantity = "qty"
        case price
        case pnl
        case pnlPct = "pnl_pct"
        case openedAt = "opened_at"
        case closedAt = "closed_at"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        symbol = try container.decode(String.self, forKey: .symbol)
        side = try container.decode(String.self, forKey: .side)
        quantity = try container.decode(Decimal.self, forKey: .quantity)
        price = try container.decode(Decimal.self, forKey: .price)
        pnl = try container.decode(Decimal.self, forKey: .pnl)
        pnlPct = try container.decode(Decimal.self, forKey: .pnlPct)
        let openedMillis = try container.decode(Double.self, forKey: .openedAt)
        openedAt = Date(timeIntervalSince1970: openedMillis / 1000)
        if let closedMillis = try container.decodeIfPresent(Double.self, forKey: .closedAt) {
            closedAt = Date(timeIntervalSince1970: closedMillis / 1000)
        } else {
            closedAt = nil
        }
    }

    func toEntity(timeZone: TimeZone) -> Trade {
        Trade(
            id: id,
            symbol: symbol,
            side: side,
            quantity: quantity,
            price: price,
            pnl: pnl,
            pnlPercent: pnlPct,
            openedAt: openedAt,
            closedAt: closedAt,
            timeZone: timeZone
        )
    }
}
