import Foundation

struct Trade: Codable, Identifiable, Equatable {
    enum Side: String, Codable {
        case buy
        case sell
        case short
        case cover

        var displayName: String { rawValue.uppercased() }
    }

    let id: String
    let symbol: String
    let side: Side
    let quantity: Decimal
    let price: Decimal
    let pnl: Decimal
    let pnlPercent: Decimal
    let openedAt: Date
    let closedAt: Date?
    let timeZoneIdentifier: String

    var timeZone: TimeZone { TimeZone(identifier: timeZoneIdentifier) ?? AppConfig.centralTimeZone }

    var isWin: Bool { pnl >= 0 }

    var closedDateInTimeZone: Date? {
        closedAt?.startOfDay(in: timeZone)
    }

    init(
        id: String,
        symbol: String,
        side: String,
        quantity: Decimal,
        price: Decimal,
        pnl: Decimal,
        pnlPercent: Decimal,
        openedAt: Date,
        closedAt: Date?,
        timeZone: TimeZone
    ) {
        self.id = id
        self.symbol = symbol
        self.side = Side(rawValue: side) ?? .buy
        self.quantity = quantity
        self.price = price
        self.pnl = pnl
        self.pnlPercent = pnlPercent
        self.openedAt = openedAt
        self.closedAt = closedAt
        self.timeZoneIdentifier = timeZone.identifier
    }
}
