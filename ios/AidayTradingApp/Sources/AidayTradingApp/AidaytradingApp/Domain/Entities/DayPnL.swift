import Foundation

struct DayPnL: Codable, Identifiable, Equatable {
    let date: Date
    let pnlAbs: Decimal

    var id: Date { date }
}
