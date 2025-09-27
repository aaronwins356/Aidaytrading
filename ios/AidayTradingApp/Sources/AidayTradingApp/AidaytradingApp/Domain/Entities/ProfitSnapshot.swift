import Foundation

struct ProfitSnapshot: Codable, Equatable {
    let balance: Decimal
    let pnlAbsolute: Decimal
    let pnlPercent: Decimal
    let winRate: Double

    var isPositive: Bool { pnlAbsolute >= 0 }
}
