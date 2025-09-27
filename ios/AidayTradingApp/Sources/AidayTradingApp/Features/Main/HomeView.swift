import Charts
import SwiftUI

struct HomeView: View {
    let context: UserSessionContext
    @StateObject private var viewModel: HomeDashboardViewModel

    init(context: UserSessionContext, reportingService: ReportingServiceProtocol) {
        self.context = context
        _viewModel = StateObject(wrappedValue: HomeDashboardViewModel(accessToken: context.tokens.accessToken, reportingService: reportingService))
    }

    init(context: UserSessionContext, viewModel: HomeDashboardViewModel) {
        self.context = context
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    equitySection
                    metricsSection
                    statusSection
                }
                .padding()
            }
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Equity & PnL")
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(Theme.background, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .task {
                viewModel.start()
            }
            .onDisappear {
                viewModel.stop()
            }
            .overlay(alignment: .bottom) {
                if let error = viewModel.errorMessage {
                    errorBanner(message: error)
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private var equitySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("Equity Curve", systemImage: "waveform.path.ecg")
                    .font(.title3.weight(.bold))
                Spacer()
                if viewModel.isLoading {
                    ProgressView()
                        .tint(.white)
                }
            }
            .foregroundStyle(.white)

            if viewModel.equitySeries.isEmpty {
                Text("No equity data available yet.")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.7))
            } else {
                Chart(viewModel.equitySeries) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Equity", point.equity.doubleValue)
                    )
                    .interpolationMethod(.monotone)
                    .foregroundStyle(.linearGradient(
                        Gradient(colors: [Theme.accentGreen.opacity(0.9), Theme.accentGreen.opacity(0.3)]),
                        startPoint: .leading,
                        endPoint: .trailing
                    ))

                    AreaMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Equity", point.equity.doubleValue)
                    )
                    .interpolationMethod(.monotone)
                    .foregroundStyle(.linearGradient(
                        Gradient(colors: [Theme.accentGreen.opacity(0.35), Theme.accentGreen.opacity(0.05)]),
                        startPoint: .top,
                        endPoint: .bottom
                    ))
                }
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 4)) { value in
                        AxisGridLine().foregroundStyle(.white.opacity(0.1))
                        AxisValueLabel(format: .dateTime.hour().minute(), centered: true)
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
                .chartYAxis {
                    AxisMarks(values: .automatic(desiredCount: 4)) { value in
                        AxisGridLine().foregroundStyle(.white.opacity(0.1))
                        if let doubleValue = value.as(Double.self) {
                            let formatted = DashboardFormatters.currency.string(from: NSNumber(value: doubleValue)) ?? ""
                            AxisValueLabel(formatted)
                                .foregroundStyle(.white.opacity(0.7))
                        }
                    }
                }
                .frame(minHeight: 220)
                .dashboardCardStyle()
            }
        }
    }

    private var metricsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Performance Snapshot")
                .font(.title3.weight(.bold))
                .foregroundStyle(.white)

            HStack(spacing: 16) {
                metricCard(title: "Balance", value: formattedCurrency(viewModel.balance ?? viewModel.profitSummary?.currentBalance))
                metricCard(title: "P/L", value: profitValueText)
                metricCard(title: "Win Rate", value: winRateText)
            }
        }
    }

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Trading Status")
                .font(.title3.weight(.bold))
                .foregroundStyle(.white)
            HStack {
                if let status = viewModel.systemStatus {
                    Label(status.running ? "Running" : "Stopped", systemImage: status.running ? "play.fill" : "pause.fill")
                        .font(.headline)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 16)
                        .background(status.running ? Theme.accentGreen.opacity(0.2) : Theme.accentRed.opacity(0.2))
                        .foregroundStyle(status.running ? Theme.accentGreen : Theme.accentRed)
                        .clipShape(Capsule())
                    if status.uptimeSeconds > 0 {
                        Text("Uptime: \(formatDuration(status.uptimeSeconds))")
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.7))
                    }
                } else if viewModel.isLoading {
                    ProgressView()
                        .tint(.white)
                } else {
                    Text("Status unavailable")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.6))
                }
                Spacer()
            }
            .dashboardCardStyle()
        }
    }

    private func metricCard(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.6))
            Text(value)
                .font(.title2.weight(.bold))
                .foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dashboardCardStyle()
    }

    private var profitValueText: String {
        guard let summary = viewModel.profitSummary else { return "—" }
        let amount = formattedCurrency(summary.totalPLAmount)
        let percent = DashboardFormatters.percent.string(from: NSNumber(value: summary.totalPLPercent.doubleValue / 100)) ?? ""
        return "\(amount) (\(percent))"
    }

    private var winRateText: String {
        guard let summary = viewModel.profitSummary else { return "—" }
        let formatted = DashboardFormatters.percent.string(from: NSNumber(value: summary.winRate)) ?? ""
        return formatted
    }

    private func formattedCurrency(_ value: Decimal?) -> String {
        guard let value else { return "—" }
        return DashboardFormatters.currency.string(from: NSDecimalNumber(decimal: value)) ?? "—"
    }

    private func formatDuration(_ seconds: Int) -> String {
        let hours = seconds / 3600
        let minutes = (seconds % 3600) / 60
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        }
        return "\(minutes)m"
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
