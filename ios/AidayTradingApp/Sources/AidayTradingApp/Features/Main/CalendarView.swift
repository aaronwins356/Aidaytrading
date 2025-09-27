import SwiftUI

struct CalendarView: View {
    let context: UserSessionContext
    @StateObject private var viewModel: CalendarDashboardViewModel

    init(context: UserSessionContext, reportingService: ReportingServiceProtocol) {
        self.context = context
        _viewModel = StateObject(
            wrappedValue: CalendarDashboardViewModel(
                month: Date(),
                accessToken: context.tokens.accessToken,
                reportingService: reportingService
            )
        )
    }

    init(context: UserSessionContext, viewModel: CalendarDashboardViewModel) {
        self.context = context
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                header
                calendarGrid
            }
            .padding()
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("PnL Calendar")
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(Theme.background, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .task {
                await viewModel.loadMonth()
            }
            .onChange(of: viewModel.month) { _, newMonth in
                Task {
                    await viewModel.loadMonth()
                }
            }
            .sheet(item: $viewModel.selectedDay) { day in
                TradeListSheet(day: day)
            }
            .overlay(alignment: .bottom) {
                if let error = viewModel.errorMessage {
                    errorBanner(message: error)
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private var header: some View {
        HStack {
            Button {
                viewModel.goToPreviousMonth()
            } label: {
                Image(systemName: "chevron.left")
                    .font(.headline)
                    .padding(8)
                    .background(Theme.cardBackground.opacity(0.6), in: Circle())
            }

            Spacer()

            Text(DateFormatter.monthYear.string(from: viewModel.month))
                .font(.title2.weight(.bold))
                .foregroundStyle(.white)
                .contentTransition(.opacity)

            Spacer()

            Button {
                viewModel.goToNextMonth()
            } label: {
                Image(systemName: "chevron.right")
                    .font(.headline)
                    .padding(8)
                    .background(Theme.cardBackground.opacity(0.6), in: Circle())
            }
        }
    }

    private var calendarGrid: some View {
        VStack(spacing: 12) {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 7), spacing: 8) {
                ForEach(Calendar.current.shortWeekdaySymbols, id: \.self) { symbol in
                    Text(symbol)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white.opacity(0.7))
                }
                if viewModel.isLoading {
                    ForEach(0..<42, id: \.self) { _ in
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Theme.cardBackground.opacity(0.4))
                            .frame(height: 44)
                            .redacted(reason: .placeholder)
                    }
                } else {
                    ForEach(viewModel.dayCells) { cell in
                        DayCellView(cell: cell)
                            .onTapGesture {
                                guard cell.date != nil, !cell.trades.isEmpty else { return }
                                viewModel.selectedDay = cell
                            }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func errorBanner(message: String) -> some View {
        Text(message)
            .font(.footnote)
            .padding()
            .frame(maxWidth: .infinity)
            .background(Theme.accentRed.opacity(0.9))
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .padding()
    }
}

private struct DayCellView: View {
    let cell: CalendarDayCell
    private let formatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 0
        formatter.currencyCode = "USD"
        return formatter
    }()

    var body: some View {
        VStack(spacing: 6) {
            if let date = cell.date {
                Text(DateFormatter.dayFormatter.string(from: date))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white)
                if let pnl = cell.pnl {
                    Text(formatter.string(from: NSDecimalNumber(decimal: pnl)) ?? "")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(color(for: pnl))
                } else {
                    Text("—")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.4))
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(backgroundColor)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(borderColor, lineWidth: 1)
        )
    }

    private var backgroundColor: Color {
        guard let pnl = cell.pnl, cell.date != nil else {
            return Theme.cardBackground.opacity(0.35)
        }
        let base = color(for: pnl)
        return base.opacity(0.15)
    }

    private var borderColor: Color {
        guard let pnl = cell.pnl, cell.date != nil else {
            return Color.white.opacity(0.1)
        }
        return color(for: pnl).opacity(0.4)
    }

    private func color(for pnl: Decimal) -> Color {
        if pnl > 0 {
            return Theme.accentGreen
        } else if pnl < 0 {
            return Theme.accentRed
        }
        return .white.opacity(0.7)
    }
}

private struct TradeListSheet: View {
    let day: CalendarDayCell
    @Environment(\.dismiss) private var dismiss
    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    var body: some View {
        NavigationStack {
            List(day.trades) { trade in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(trade.symbol)
                            .font(.headline)
                        Spacer()
                        Text(trade.pnl >= 0 ? "Win" : "Loss")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(trade.pnl >= 0 ? Theme.accentGreen : Theme.accentRed)
                    }
                    Text("Side: \(trade.side.rawValue.uppercased())")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Size: \(trade.size.doubleValue, format: .number.precision(.fractionLength(2)))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(dateFormatter.string(from: trade.timestamp))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }
            .navigationTitle(day.date.map { DateFormatter.mediumFormatter.string(from: $0) } ?? "Trades")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

private extension DateFormatter {
    static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "d"
        return formatter
    }()

    static let mediumFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        return formatter
    }()

    static let monthYear: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "LLLL yyyy"
        return formatter
    }()
}
