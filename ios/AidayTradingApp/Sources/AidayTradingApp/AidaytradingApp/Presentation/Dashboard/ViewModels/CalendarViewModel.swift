import Foundation

@MainActor
final class CalendarViewModel: ObservableObject {
    struct CalendarCellModel: Identifiable {
        enum Sentiment {
            case positive
            case negative
            case neutral
        }

        let id = UUID()
        let date: Date
        let pnl: Decimal?
        let isCurrentMonth: Bool

        var sentiment: Sentiment {
            guard let pnl else { return .neutral }
            if pnl > 0 { return .positive }
            if pnl < 0 { return .negative }
            return .neutral
        }
    }

    @Published var isLoading = false
    @Published var days: [DayPnL] = []
    @Published var selectedMonth: Date
    @Published var monthPnLTotal: Decimal = 0
    @Published var error: String?
    @Published var cellModels: [CalendarCellModel] = []

    private let repository: TradesRepository
    private let timeZone: TimeZone
    private var allTrades: [Trade] = []
    private var allDayPnL: [DayPnL] = []

    init(repository: TradesRepository = TradesRepositoryImpl(), timeZone: TimeZone = AppConfig.centralTimeZone) {
        self.repository = repository
        self.timeZone = timeZone
        self.selectedMonth = Date().startOfMonth(in: timeZone)
    }

    func loadMonth(_ date: Date) async {
        selectedMonth = date.startOfMonth(in: timeZone)
        isLoading = true
        error = nil
        do {
            allTrades = try await repository.fetchRecentTrades(limit: 500)
            allDayPnL = CalendarPnLComputer.computeDailyPnL(trades: allTrades, in: timeZone)
            updateVisibleMonth()
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func nextMonth() {
        selectedMonth = selectedMonth.adding(days: 32, in: timeZone).startOfMonth(in: timeZone)
        updateVisibleMonth()
    }

    func previousMonth() {
        selectedMonth = selectedMonth.adding(days: -1, in: timeZone).startOfMonth(in: timeZone)
        updateVisibleMonth()
    }

    func trades(on date: Date) -> [Trade] {
        let targetDay = date.startOfDay(in: timeZone)
        return allTrades.filter { trade in
            trade.closedDateInTimeZone == targetDay
        }
    }

    private func updateVisibleMonth() {
        let start = selectedMonth
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        guard let range = calendar.range(of: .day, in: .month, for: start) else { return }
        let numberOfDays = range.count
        let firstWeekday = calendar.component(.weekday, from: start)
        let leading = (firstWeekday - calendar.firstWeekday + 7) % 7
        var dates: [Date] = []
        if leading > 0 {
            for offset in stride(from: leading, to: 0, by: -1) {
                if let date = calendar.date(byAdding: .day, value: -offset, to: start) {
                    dates.append(date)
                }
            }
        }
        for day in 0..<numberOfDays {
            if let date = calendar.date(byAdding: .day, value: day, to: start) {
                dates.append(date)
            }
        }
        while dates.count % 7 != 0 {
            if let last = dates.last, let next = calendar.date(byAdding: .day, value: 1, to: last) {
                dates.append(next)
            } else {
                break
            }
        }
        let pnlDictionary = Dictionary(uniqueKeysWithValues: allDayPnL.map { ($0.date, $0.pnlAbs) })
        days = allDayPnL.filter { day in
            calendar.isDate(day.date, equalTo: start, toGranularity: .month)
        }
        monthPnLTotal = days.reduce(0) { $0 + $1.pnlAbs }
        cellModels = dates.map { date in
            let startOfDay = date.startOfDay(in: timeZone)
            let pnl = pnlDictionary[startOfDay]
            let isCurrent = calendar.isDate(date, equalTo: start, toGranularity: .month)
            return CalendarCellModel(date: date, pnl: pnl, isCurrentMonth: isCurrent)
        }
    }
}
