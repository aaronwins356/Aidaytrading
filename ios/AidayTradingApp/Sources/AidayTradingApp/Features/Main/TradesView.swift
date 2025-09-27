import SwiftUI

struct TradesView: View {
    let context: UserSessionContext
    @StateObject private var viewModel: TradesViewModel
    @State private var symbolDebounceTask: Task<Void, Never>?

    init(context: UserSessionContext, reportingService: ReportingServiceProtocol) {
        self.context = context
        _viewModel = StateObject(wrappedValue: TradesViewModel(accessToken: context.tokens.accessToken, reportingService: reportingService))
    }

    init(context: UserSessionContext, viewModel: TradesViewModel) {
        self.context = context
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                filterControls
                tradesList
            }
            .padding()
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Trades")
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(Theme.background, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .task {
                await viewModel.loadInitial()
            }
            .onChange(of: viewModel.symbolFilter) { _, _ in
                symbolDebounceTask?.cancel()
                symbolDebounceTask = Task { [symbol = viewModel.symbolFilter] in
                    try? await Task.sleep(nanoseconds: 350_000_000)
                    await MainActor.run {
                        if symbol == viewModel.symbolFilter {
                            Task { await viewModel.loadInitial() }
                        }
                    }
                }
            }
            .onChange(of: viewModel.outcomeFilter) { _, _ in
                Task { await viewModel.loadInitial() }
            }
            .onDisappear {
                viewModel.stop()
            }
            .overlay(alignment: .bottom) {
                if viewModel.realtimeWarningMessage != nil || viewModel.errorMessage != nil {
                    VStack(spacing: 8) {
                        if let realtimeWarning = viewModel.realtimeWarningMessage {
                            errorBanner(message: realtimeWarning)
                        }
                        if let error = viewModel.errorMessage {
                            errorBanner(message: error)
                        }
                    }
                    .padding(.bottom, 16)
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private var filterControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Filters")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.6))
            HStack(spacing: 12) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.white.opacity(0.6))
                    TextField("Symbol", text: $viewModel.symbolFilter)
                        .textInputAutocapitalization(.characters)
                        .disableAutocorrection(true)
                        .foregroundStyle(.white)
                }
                .padding(12)
                .background(Theme.cardBackground.cornerRadius(12))

                Picker("Outcome", selection: $viewModel.outcomeFilter) {
                    ForEach(TradeOutcomeFilter.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .background(Theme.cardBackground.opacity(0.4).cornerRadius(12))
            }
        }
    }

    private var tradesList: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                ForEach(viewModel.filteredTrades) { trade in
                    TradeRow(trade: trade)
                        .onAppear {
                            if trade.id == viewModel.filteredTrades.last?.id {
                                Task { await viewModel.loadMore() }
                            }
                        }
                }

                if viewModel.isLoading {
                    ProgressView()
                        .tint(.white)
                        .padding()
                } else if viewModel.filteredTrades.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "tray")
                            .font(.system(size: 48))
                            .foregroundStyle(.white.opacity(0.4))
                        Text("No trades match the selected filters yet.")
                            .font(.footnote)
                            .foregroundStyle(.white.opacity(0.6))
                    }
                    .padding(.top, 32)
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

private struct TradeRow: View {
    let trade: TradeRecord
    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(trade.symbol)
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(.white)
                    Text(dateFormatter.string(from: trade.timestamp))
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                }
                Spacer()
                Text(trade.pnl >= 0 ? "Win" : "Loss")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(trade.pnl >= 0 ? Theme.accentGreen : Theme.accentRed)
                    .padding(.vertical, 4)
                    .padding(.horizontal, 8)
                    .background((trade.pnl >= 0 ? Theme.accentGreen : Theme.accentRed).opacity(0.15), in: Capsule())
            }

            HStack {
                Label("Side: \(trade.side.rawValue.uppercased())", systemImage: "arrow.left.arrow.right")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
                Spacer()
                Label(
                    "Size: \(trade.size.doubleValue, format: .number.precision(.fractionLength(2)))",
                    systemImage: "chart.bar.xaxis"
                )
                .font(.caption)
                .foregroundStyle(.white.opacity(0.7))
                Label(
                    "P/L: \(DashboardFormatters.currency.string(from: NSDecimalNumber(decimal: trade.pnl)) ?? "—")",
                    systemImage: trade.pnl >= 0 ? "arrowtriangle.up.fill" : "arrowtriangle.down.fill"
                )
                .font(.caption)
                .foregroundStyle(trade.pnl >= 0 ? Theme.accentGreen : Theme.accentRed)
            }
        }
        .padding()
        .background(Theme.cardBackground.cornerRadius(16))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.white.opacity(0.06), lineWidth: 1)
        )
    }
}
