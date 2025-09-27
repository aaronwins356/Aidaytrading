import Foundation
import SwiftUI

struct CalendarDayCell: Identifiable {
    let id = UUID()
    let date: Date?
    let isCurrentMonth: Bool
    let pnl: Decimal?
    let trades: [TradeRecord]
}

@MainActor
final class CalendarDashboardViewModel: ObservableObject {
    @Published var month: Date
    @Published var dayCells: [CalendarDayCell] = []
    @Published var selectedDay: CalendarDayCell?
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let accessToken: String
    private let reportingService: ReportingServiceProtocol
    private let calendar: Calendar
    private let centralTimeZone = TimeZone(identifier: "America/Chicago")!

    init(month: Date = Date(), accessToken: String, reportingService: ReportingServiceProtocol) {
        self.month = month
        self.accessToken = accessToken
        self.reportingService = reportingService
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = centralTimeZone
        self.calendar = calendar
    }

    func loadMonth() async {
        isLoading = true
        errorMessage = nil
        selectedDay = nil
        do {
            let range = monthBoundary(for: month)
            var trades: [TradeRecord] = []
            var page = 1
            let pageSize = 200
            while true {
                let pageResponse = try await reportingService.fetchTrades(
                    accessToken: accessToken,
                    page: page,
                    pageSize: pageSize,
                    symbol: nil,
                    side: nil,
                    start: range.start,
                    end: range.end
                )
                trades.append(contentsOf: pageResponse.items)
                if trades.count >= pageResponse.total || pageResponse.items.count < pageSize {
                    break
                }
                page += 1
            }
            let grouped = groupTradesByDay(trades: trades)
            dayCells = buildCalendarCells(range: range, grouped: grouped)
        } catch {
            errorMessage = (error as? APIErrorResponse)?.message ?? error.localizedDescription
        }
        isLoading = false
    }

    func goToPreviousMonth() {
        if let newMonth = calendar.date(byAdding: .month, value: -1, to: month) {
            month = newMonth
        }
    }

    func goToNextMonth() {
        if let newMonth = calendar.date(byAdding: .month, value: 1, to: month) {
            month = newMonth
        }
    }

    private func monthBoundary(for date: Date) -> (start: Date, end: Date) {
        let startOfMonth = calendar.date(from: calendar.dateComponents([.year, .month], from: date))!
        let startOfNextMonth = calendar.date(byAdding: DateComponents(month: 1), to: startOfMonth)!
        return (startOfMonth, startOfNextMonth)
    }

    private func groupTradesByDay(trades: [TradeRecord]) -> [Date: (pnl: Decimal, trades: [TradeRecord])] {
        trades.reduce(into: [:]) { partialResult, trade in
            let day = calendar.startOfDay(for: trade.timestamp)
            var entry = partialResult[day] ?? (pnl: 0, trades: [])
            entry.pnl += trade.pnl
            entry.trades.append(trade)
            partialResult[day] = entry
        }
    }

    private func buildCalendarCells(
        range: (start: Date, end: Date),
        grouped: [Date: (pnl: Decimal, trades: [TradeRecord])]
    ) -> [CalendarDayCell] {
        let daysInMonth = calendar.range(of: .day, in: .month, for: range.start) ?? 1..<31
        let firstWeekday = calendar.component(.weekday, from: range.start)
        let leadingEmpty = (firstWeekday + 6) % 7
        var cells: [CalendarDayCell] = []
        cells.reserveCapacity(leadingEmpty + daysInMonth.count)

        if leadingEmpty > 0 {
            for _ in 0..<leadingEmpty {
                cells.append(CalendarDayCell(date: nil, isCurrentMonth: false, pnl: nil, trades: []))
            }
        }

        for day in daysInMonth {
            guard let date = calendar.date(byAdding: .day, value: day - 1, to: range.start) else { continue }
            let dayKey = calendar.startOfDay(for: date)
            let payload = grouped[dayKey]
            cells.append(
                CalendarDayCell(
                    date: date,
                    isCurrentMonth: true,
                    pnl: payload?.pnl,
                    trades: payload?.trades.sorted(by: { $0.timestamp < $1.timestamp }) ?? []
                )
            )
        }

        let totalCells = cells.count
        let remainder = totalCells % 7
        if remainder != 0 {
            for _ in 0..<(7 - remainder) {
                cells.append(CalendarDayCell(date: nil, isCurrentMonth: false, pnl: nil, trades: []))
            }
        }
        return cells
    }
}
