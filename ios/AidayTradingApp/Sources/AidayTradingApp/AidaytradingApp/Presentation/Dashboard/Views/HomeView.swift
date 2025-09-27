import SwiftUI

struct HomeView: View {
    @StateObject private var viewModel: HomeViewModel
    @EnvironmentObject private var realtimeClient: TradingWebSocketClient

    init(viewModel: HomeViewModel = HomeViewModel()) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    header
                    statusSection
                    metricsSection
                    EquityChartView(points: viewModel.equity, delta: viewModel.equityDelta())
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 24)
            }
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Dashboard")
            .toolbarBackground(Theme.background, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .task {
                await viewModel.loadInitial()
            }
            .onAppear {
                viewModel.attachRealtime(realtimeClient)
            }
            .overlay(alignment: .top) {
                if viewModel.isStale {
                    staleBanner
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 8) {
                Text("Dashboard")
                    .font(Theme.Typography.headline)
                    .foregroundColor(Theme.primaryText)
                    .accessibilityAddTraits(.isHeader)
                if let lastUpdated = viewModel.lastUpdated {
                    Text("Updated \(lastUpdated.formatted(date: .omitted, time: .shortened))")
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Theme.cardBackground)
                        .foregroundColor(Theme.secondaryText)
                        .clipShape(Capsule())
                        .accessibilityLabel("Last updated at \(lastUpdated.formatted(date: .omitted, time: .shortened))")
                }
            }
            Spacer()
        }
    }

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Status")
            if let status = viewModel.status {
                StatusBadge(isRunning: status.running, uptime: status.formattedUptime.isEmpty ? nil : status.formattedUptime)
            } else if viewModel.isLoading {
                StatusBadge(isRunning: true, uptime: "")
                    .skeleton()
            } else if let error = viewModel.error {
                ErrorStateView(title: "Status unavailable", message: error, retryTitle: "Retry") {
                    Task { await viewModel.loadInitial() }
                }
            }
        }
    }

    private var metricsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader(title: "Performance")
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                MetricCard(
                    title: "Balance",
                    value: viewModel.profit?.balance.currencyString() ?? "—",
                    subtitle: "USD"
                )
                MetricCard(
                    title: "P/L",
                    value: viewModel.profit?.pnlAbsolute.currencyString() ?? "—",
                    subtitle: viewModel.profit.map { $0.pnlPercent.signedPercentString() } ?? "—",
                    tint: (viewModel.profit?.isPositive ?? true) ? .positive : .negative
                )
                MetricCard(
                    title: "Win Rate",
                    value: winRateText
                )
                MetricCard(
                    title: "Equity Points",
                    value: "\(viewModel.equity.count)",
                    subtitle: "Snapshots"
                )
            }
            .accessibilityElement(children: .contain)
        }
        .modifier(LoadingOverlay(isLoading: viewModel.isLoading && viewModel.profit == nil))
    }

    private var staleBanner: some View {
        Text("Data may be stale. Check your connection.")
            .font(.footnote)
            .foregroundColor(.black)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Theme.accentRed)
            .clipShape(Capsule())
            .padding(.top, 16)
            .accessibilityLabel("Warning: data may be stale")
    }

    private var winRateText: String {
        guard let winRate = viewModel.profit?.winRate else { return "—" }
        let formatter = NumberFormatter()
        formatter.locale = Locale.current
        formatter.numberStyle = .percent
        formatter.maximumFractionDigits = 1
        formatter.minimumFractionDigits = 1
        formatter.positivePrefix = ""
        formatter.negativePrefix = ""
        return formatter.string(from: NSNumber(value: winRate)) ?? "—"
    }
}

private struct LoadingOverlay: ViewModifier {
    let isLoading: Bool

    func body(content: Content) -> some View {
        ZStack(alignment: .center) {
            content
            if isLoading {
                LoadingStateView()
            }
        }
    }
}
