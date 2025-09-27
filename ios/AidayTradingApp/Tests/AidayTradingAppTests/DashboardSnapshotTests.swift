import SwiftUI
import XCTest
@testable import AidayTradingApp

#if canImport(SwiftUI)

@MainActor
final class DashboardSnapshotTests: XCTestCase {
    private let context = UserSessionContext(
        profile: UserProfile(
            id: UUID(),
            username: "snapshot",
            email: "snapshot@example.com",
            role: .viewer,
            approvalStatus: .approved
        ),
        tokens: AuthTokens(accessToken: "token", refreshToken: "refresh", accessTokenExpiry: .distantFuture)
    )

    func testHomeDashboardSnapshotProducesImage() async throws {
        let service = MockReportingService()
        service.statusResult = .success(SystemStatus(running: true, uptimeSeconds: 7200))
        service.profitResult = .success(
            ProfitSummary(
                currentBalance: Decimal(string: "10500")!,
                totalPLAmount: Decimal(string: "500")!,
                totalPLPercent: Decimal(string: "5.0")!,
                winRate: 0.58
            )
        )
        service.balanceResult = .success(Decimal(string: "10500")!)
        service.equityResult = .success([
            EquityCurvePoint(timestamp: Date().addingTimeInterval(-3600 * 3), equity: Decimal(string: "9800")!),
            EquityCurvePoint(timestamp: Date().addingTimeInterval(-3600 * 2), equity: Decimal(string: "10050")!),
            EquityCurvePoint(timestamp: Date().addingTimeInterval(-3600), equity: Decimal(string: "10200")!),
            EquityCurvePoint(timestamp: Date(), equity: Decimal(string: "10500")!)
        ])

        let viewModel = HomeDashboardViewModel(
            accessToken: context.tokens.accessToken,
            reportingService: service,
            realtimeService: MockRealtimeService(),
            notificationScheduler: MockLocalNotificationScheduler()
        )
        await viewModel.loadData()

        let view = HomeView(context: context, viewModel: viewModel)
        let renderer = ImageRenderer(content: view.frame(width: 375, height: 812))

        #if canImport(UIKit)
        XCTAssertNotNil(renderer.uiImage)
        #elseif canImport(AppKit)
        XCTAssertNotNil(renderer.nsImage)
        #else
        throw XCTSkip("Snapshot rendering not supported on this platform")
        #endif
    }

    func testCalendarSnapshotProducesImage() async throws {
        let service = MockReportingService()
        let formatter = ISO8601DateFormatter.apiFormatter
        let trades = [
            TradeRecord(
                id: 1,
                symbol: "BTCUSD",
                side: .buy,
                size: Decimal(string: "0.5")!,
                pnl: Decimal(string: "200.0")!,
                timestamp: formatter.date(from: "2024-05-10T15:30:00.000Z")!
            ),
            TradeRecord(
                id: 2,
                symbol: "BTCUSD",
                side: .sell,
                size: Decimal(string: "0.25")!,
                pnl: Decimal(string: "-80.0")!,
                timestamp: formatter.date(from: "2024-05-11T18:45:00.000Z")!
            )
        ]
        service.tradesResult = .success(TradesPage(items: trades, page: 1, pageSize: 200, total: trades.count))

        let viewModel = CalendarDashboardViewModel(month: formatter.date(from: "2024-05-01T00:00:00.000Z")!, accessToken: context.tokens.accessToken, reportingService: service)
        await viewModel.loadMonth()

        let view = CalendarView(context: context, viewModel: viewModel)
        let renderer = ImageRenderer(content: view.frame(width: 375, height: 700))

        #if canImport(UIKit)
        XCTAssertNotNil(renderer.uiImage)
        #elseif canImport(AppKit)
        XCTAssertNotNil(renderer.nsImage)
        #else
        throw XCTSkip("Snapshot rendering not supported on this platform")
        #endif
    }
}
#endif
