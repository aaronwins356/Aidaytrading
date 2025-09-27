import Foundation
import SwiftUI

@MainActor
final class HomeDashboardViewModel: ObservableObject {
    @Published var equitySeries: [EquityCurvePoint] = []
    @Published var profitSummary: ProfitSummary?
    @Published var systemStatus: SystemStatus?
    @Published var balance: Decimal?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var realtimeWarningMessage: String?

    private let accessToken: String
    private let reportingService: ReportingServiceProtocol
    private var refreshTask: Task<Void, Never>?
    private let realtimeService: RealtimeServiceProtocol
    private let notificationScheduler: LocalNotificationScheduling

    init(
        accessToken: String,
        reportingService: ReportingServiceProtocol,
        realtimeService: RealtimeServiceProtocol = RealtimeService(),
        notificationScheduler: LocalNotificationScheduling = LocalNotificationScheduler()
    ) {
        self.accessToken = accessToken
        self.reportingService = reportingService
        self.realtimeService = realtimeService
        self.notificationScheduler = notificationScheduler
        self.realtimeService.delegate = self
    }

    func start() {
        guard refreshTask == nil else { return }
        realtimeService.connect(accessToken: accessToken)
        refreshTask = Task { [weak self] in
            await self?.loadData()
            while !(Task.isCancelled) {
                try? await Task.sleep(nanoseconds: 600 * 1_000_000_000)
                await self?.loadData()
            }
        }
    }

    func stop() {
        refreshTask?.cancel()
        refreshTask = nil
        realtimeService.disconnect()
        notificationScheduler.cancelRealtimeStallNotification()
    }

    deinit {
        refreshTask?.cancel()
    }

    func loadData() async {
        isLoading = true
        errorMessage = nil
        do {
            async let statusTask: SystemStatus = try await reportingService.fetchSystemStatus(accessToken: accessToken)
            async let equityTask: [EquityCurvePoint] = try await reportingService.fetchEquityCurve(accessToken: accessToken, start: nil, end: nil, limit: 120)
            async let balanceTask: Decimal = try await reportingService.fetchCurrentBalance(accessToken: accessToken)

            let profit = try await reportingService.fetchProfitSummary(accessToken: accessToken)
            let status = try await statusTask
            let equity = try await equityTask
            let balanceValue = (try? await balanceTask) ?? profit.currentBalance
            systemStatus = status
            profitSummary = profit
            balance = balanceValue
            equitySeries = equity.sorted(by: { $0.timestamp < $1.timestamp })
        } catch {
            errorMessage = (error as? APIErrorResponse)?.message ?? error.localizedDescription
        }
        isLoading = false
    }
}

extension HomeDashboardViewModel: RealtimeServiceDelegate {
    func realtimeServiceDidConnect(_ service: RealtimeServiceProtocol) {
        realtimeWarningMessage = nil
        notificationScheduler.cancelRealtimeStallNotification()
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didDisconnectWith error: Error?) {
        realtimeWarningMessage = "Realtime feed disconnected."
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveStatus status: SystemStatus) {
        systemStatus = status
        notificationScheduler.cancelRealtimeStallNotification()
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveEquityPoint point: EquityCurvePoint) {
        if !equitySeries.contains(where: { $0.timestamp == point.timestamp }) {
            equitySeries.append(point)
            equitySeries.sort(by: { $0.timestamp < $1.timestamp })
        }
        notificationScheduler.cancelRealtimeStallNotification()
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveTrade trade: TradeRecord) {
        // Home dashboard does not display trade level data.
    }

    func realtimeServiceDidDetectStall(_ service: RealtimeServiceProtocol) {
        realtimeWarningMessage = "Realtime feed disconnected."
        notificationScheduler.cancelRealtimeStallNotification()
        Task {
            await notificationScheduler.scheduleRealtimeStallNotification()
        }
    }
}
