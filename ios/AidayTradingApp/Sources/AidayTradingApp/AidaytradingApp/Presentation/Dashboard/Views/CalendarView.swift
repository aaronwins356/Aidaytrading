import SwiftUI

struct CalendarView: View {
    @StateObject private var viewModel: CalendarViewModel
    @State private var activeDay: CalendarViewModel.CalendarCellModel?

    init(viewModel: CalendarViewModel = CalendarViewModel()) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    weekdayRow
                    calendarGrid
                    summarySection
                }
                .padding(20)
            }
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Calendar")
            .toolbarBackground(Theme.background, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .task {
                await viewModel.loadMonth(Date())
            }
            .sheet(item: $activeDay) { model in
                DayDetailView(date: model.date, trades: viewModel.trades(on: model.date))
                    .presentationDetents([.medium, .large])
                    .presentationBackground(Theme.background)
            }
        }
        .preferredColorScheme(.dark)
    }

    private var header: some View {
        HStack {
            Button(action: viewModel.previousMonth) {
                Image(systemName: "chevron.left")
            }
            .buttonStyle(.plain)
            .foregroundColor(Theme.primaryText)
            Spacer()
            Text(viewModel.selectedMonth.formatted(.dateTime.month(.wide).year()))
                .font(Theme.Typography.headline)
                .foregroundColor(Theme.primaryText)
                .accessibilityAddTraits(.isHeader)
            Spacer()
            Button(action: viewModel.nextMonth) {
                Image(systemName: "chevron.right")
            }
            .buttonStyle(.plain)
            .foregroundColor(Theme.primaryText)
        }
    }

    private var weekdayRow: some View {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = AppConfig.centralTimeZone
        let symbols = calendar.shortStandaloneWeekdaySymbols
        let firstWeekdayIndex = calendar.firstWeekday - 1
        let weekdays = Array(symbols[firstWeekdayIndex...]) + Array(symbols[..<firstWeekdayIndex])
        return HStack {
            ForEach(weekdays, id: \.self) { day in
                Text(day.uppercased())
                    .font(.caption)
                    .foregroundColor(Theme.secondaryText)
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private var calendarGrid: some View {
        Group {
            if viewModel.isLoading {
                LoadingStateView()
            } else if let error = viewModel.error {
                ErrorStateView(title: "Failed to load", message: error, retryTitle: "Retry") {
                    Task { await viewModel.loadMonth(viewModel.selectedMonth) }
                }
            } else if viewModel.cellModels.isEmpty {
                EmptyStateView(title: "No trades", message: "No closed trades this month.")
            } else {
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 7), spacing: 8) {
                    ForEach(viewModel.cellModels) { model in
                        dayCell(for: model)
                            .onTapGesture {
                                if model.pnl != nil {
                                    activeDay = model
                                }
                            }
                    }
                }
            }
        }
    }

    private var summarySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Monthly P/L")
                .font(.headline)
                .foregroundColor(Theme.primaryText)
            Text(viewModel.monthPnLTotal.currencyString())
                .font(Theme.Typography.metric)
                .foregroundColor(viewModel.monthPnLTotal >= 0 ? Theme.accentGreen : Theme.accentRed)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func dayCell(for model: CalendarViewModel.CalendarCellModel) -> some View {
        VStack(spacing: 6) {
            var calendar = Calendar(identifier: .gregorian)
            calendar.timeZone = AppConfig.centralTimeZone
            Text("\(calendar.component(.day, from: model.date))")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(Theme.primaryText)
            if let pnl = model.pnl {
                Text(pnl.currencyString())
                    .font(.caption)
                    .foregroundColor(Theme.primaryText)
                    .lineLimit(1)
            } else {
                Text("—")
                    .font(.caption)
                    .foregroundColor(Theme.secondaryText)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(backgroundColor(for: model))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .accessibilityLabel(accessibilityLabel(for: model))
    }

    private func backgroundColor(for model: CalendarViewModel.CalendarCellModel) -> Color {
        guard model.isCurrentMonth, let pnl = model.pnl else {
            return Theme.cardBackground.opacity(0.3)
        }
        if pnl > 0 {
            return Theme.accentGreen.opacity(0.3)
        } else if pnl < 0 {
            return Theme.accentRed.opacity(0.3)
        } else {
            return Theme.cardBackground.opacity(0.4)
        }
    }

    private func accessibilityLabel(for model: CalendarViewModel.CalendarCellModel) -> String {
        let day = model.date.formatted(.dateTime.month(.abbreviated).day())
        guard let pnl = model.pnl else { return "\(day), no data" }
        let amount = pnl.currencyString()
        let sentiment = pnl > 0 ? "Positive" : (pnl < 0 ? "Negative" : "Flat")
        return "\(day): \(amount), \(sentiment)"
    }
}

private struct DayDetailView: View {
    let date: Date
    let trades: [Trade]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section(header: Text(date.formatted(.dateTime.month(.wide).day().year()))) {
                    Text(totalPnL.currencyString())
                        .font(.title2.weight(.bold))
                        .foregroundColor(totalPnL >= 0 ? Theme.accentGreen : Theme.accentRed)
                        .accessibilityLabel("Total profit \(totalPnL.currencyString())")
                    Text("\(trades.count) trades")
                        .font(.subheadline)
                        .foregroundColor(Theme.secondaryText)
                }
                Section("Trades") {
                    if trades.isEmpty {
                        Text("No trades closed on this day.")
                            .foregroundColor(Theme.secondaryText)
                    } else {
                        ForEach(trades) { trade in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(trade.symbol)
                                        .font(.headline)
                                        .foregroundColor(Theme.primaryText)
                                    Spacer()
                                    Text(trade.side.displayName)
                                        .font(.caption.bold())
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Theme.cardBackground)
                                        .clipShape(Capsule())
                                }
                                Text(trade.pnl.currencyString())
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundColor(trade.isWin ? Theme.accentGreen : Theme.accentRed)
                                if let closed = trade.closedAt {
                                    Text(closed.formatted(.dateTime.hour(.defaultDigits).minute()))
                                        .font(.caption)
                                        .foregroundColor(Theme.secondaryText)
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
            }
            .background(Theme.background)
            .scrollContentBackground(.hidden)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
        }
    }

    private var totalPnL: Decimal {
        trades.reduce(0) { $0 + $1.pnl }
    }
}
