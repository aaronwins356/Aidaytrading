import SwiftUI

struct TradesListView: View {
    @StateObject private var viewModel: TradesViewModel
    @State private var filters: TradesViewModel.TradesFilters = .empty

    init(viewModel: TradesViewModel = TradesViewModel()) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading && viewModel.items.isEmpty {
                    LoadingStateView()
                        .padding()
                } else if let error = viewModel.error, viewModel.items.isEmpty {
                    ErrorStateView(title: "Failed to load trades", message: error, retryTitle: "Retry") {
                        Task { await viewModel.loadFirstPage() }
                    }
                    .padding()
                } else if viewModel.items.isEmpty {
                    EmptyStateView(title: "No trades", message: "Closed trades will appear here once available.")
                        .padding()
                } else {
                    listContent
                }
            }
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Trades")
            .toolbarBackground(Theme.background, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .task {
                await viewModel.loadFirstPage()
            }
            .refreshable {
                await viewModel.loadFirstPage()
            }
        }
        .preferredColorScheme(.dark)
    }

    private var listContent: some View {
        List {
            Section {
                FiltersView(symbols: viewModel.availableSymbols, filters: $filters) { filters in
                    Task { await applyFilters(filters) }
                } onReset: {
                    Task { await resetFilters() }
                }
            }
            .listRowBackground(Theme.background)
            tradesSection
            if viewModel.nextCursor != nil {
                loadingFooter
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Theme.background)
    }

    private var tradesSection: some View {
        Section(header: Text("Closed Trades").foregroundColor(Theme.secondaryText)) {
            ForEach(Array(viewModel.items.enumerated()), id: \.element.id) { index, trade in
                tradeRow(trade)
                    .onAppear {
                        Task { await viewModel.loadNextPageIfNeeded(currentIndex: index) }
                    }
            }
        }
        .listRowBackground(Theme.cardBackground)
    }

    private func tradeRow(_ trade: Trade) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(trade.symbol)
                    .font(.headline)
                    .foregroundColor(Theme.primaryText)
                Spacer()
                Text(trade.side.displayName)
                    .font(.caption.bold())
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(trade.side == .buy ? Theme.accentGreen.opacity(0.2) : Theme.cardBackground.opacity(0.5))
                    .foregroundColor(Theme.primaryText)
                    .clipShape(Capsule())
            }
            Text("Qty \(trade.quantity) @ \(trade.price.currencyString())")
                .font(.subheadline)
                .foregroundColor(Theme.secondaryText)
            HStack {
                Text(trade.pnl.currencyString())
                    .font(.headline)
                    .foregroundColor(trade.isWin ? Theme.accentGreen : Theme.accentRed)
                Spacer()
                Text(trade.pnlPercent.signedPercentString())
                    .font(.caption)
                    .foregroundColor(trade.isWin ? Theme.accentGreen : Theme.accentRed)
            }
            if let closed = trade.closedAt {
                Text(closed.formatted(.dateTime.month(.abbreviated).day().hour(.defaultDigits).minute()))
                    .font(.caption)
                    .foregroundColor(Theme.secondaryText)
            }
        }
        .padding(.vertical, 8)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(trade.symbol), \(trade.side.displayName). Size \(trade.quantity) at \(trade.price.currencyString()). P L \(trade.pnl.currencyString())")
    }

    private var loadingFooter: some View {
        HStack {
            Spacer()
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: Theme.accentGreen))
            Spacer()
        }
        .listRowBackground(Theme.background)
    }

    private func applyFilters(_ filters: TradesViewModel.TradesFilters) async {
        viewModel.apply(filters: filters)
        if !viewModel.items.isEmpty {
            await viewModel.loadNextPageIfNeeded(currentIndex: viewModel.items.count - 1)
        }
    }

    private func resetFilters() async {
        filters = .empty
        viewModel.resetFilters()
        await viewModel.loadFirstPage()
    }
}
