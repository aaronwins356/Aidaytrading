import Foundation

struct EquityPoint: Codable, Identifiable, Equatable {
    let timestamp: Date
    let equity: Decimal

    var id: Date { timestamp }
}
