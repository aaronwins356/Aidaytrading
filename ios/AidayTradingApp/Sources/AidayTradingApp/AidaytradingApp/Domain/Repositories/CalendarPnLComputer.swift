import Foundation

struct CalendarPnLComputer {
    static func computeDailyPnL(trades: [Trade], in timeZone: TimeZone) -> [DayPnL] {
        var totals: [Date: Decimal] = [:]
        for trade in trades {
            guard let closedAt = trade.closedAt else { continue }
            let day = closedAt.startOfDay(in: timeZone)
            let current = totals[day] ?? 0
            totals[day] = current + trade.pnl
        }
        return totals
            .map { DayPnL(date: $0.key, pnlAbs: $0.value) }
            .sorted { $0.date < $1.date }
    }
}
