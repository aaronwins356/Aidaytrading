import Foundation
import SwiftUI

@MainActor
final class RiskViewModel: ObservableObject {
    @Published var configuration: RiskConfiguration = .default
    @Published private(set) var isLoading = false
    @Published private(set) var isSaving = false
    @Published var successMessage: String?
    @Published var errorMessage: String?

    weak var actionRecorder: AdminActionRecording?

    private let repository: AdminRepository
    private var originalConfiguration: RiskConfiguration = .default

    init(repository: AdminRepository) {
        self.repository = repository
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let configuration = try await repository.fetchRiskConfiguration()
            self.configuration = configuration
            originalConfiguration = configuration
        } catch {
            errorMessage = "Failed to load risk configuration: \(error.localizedDescription)"
        }
    }

    func binding(for keyPath: ReferenceWritableKeyPath<RiskConfiguration, Double>, range: ClosedRange<Double>) -> Binding<Double> {
        Binding(
            get: { self.configuration[keyPath: keyPath] },
            set: { newValue in
                self.configuration[keyPath: keyPath] = range.clamped(newValue)
            }
        )
    }

    func positionsBinding() -> Binding<Int> {
        Binding(
            get: { self.configuration.maxOpenPositions },
            set: { newValue in
                let clamped = Int(RiskConfiguration.ranges.maxOpenPositions.clamped(Double(newValue)))
                self.configuration.maxOpenPositions = clamped
            }
        )
    }

    var hasChanges: Bool {
        configuration.clamped() != originalConfiguration.clamped()
    }

    func save() async {
        guard !isSaving else { return }
        let candidate = configuration.clamped()
        guard candidate.validate() else {
            errorMessage = "Risk parameters must remain within approved guardrails."
            configuration = candidate
            return
        }
        successMessage = nil
        isSaving = true
        defer { isSaving = false }
        do {
            let updated = try await repository.updateRiskConfiguration(candidate)
            configuration = updated
            originalConfiguration = updated
            successMessage = "Risk updated at \(DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .short))"
            let details = "Max DD: \(Int(updated.maxDrawdownPercent))%, Daily loss: \(Int(updated.dailyLossLimitPercent))%, Risk/trade: \(riskPerTradeText(updated.riskPerTrade)), Max positions: \(updated.maxOpenPositions)"
            actionRecorder?.recordChange(
                category: .risk,
                summary: "Risk guardrails updated",
                details: details,
                payload: [
                    "max_drawdown_percent": updated.maxDrawdownPercent,
                    "daily_loss_limit_percent": updated.dailyLossLimitPercent,
                    "risk_per_trade": updated.riskPerTrade,
                    "max_open_positions": updated.maxOpenPositions,
                    "atr_stop_loss_multiplier": updated.atrStopLossMultiplier,
                    "atr_take_profit_multiplier": updated.atrTakeProfitMultiplier
                ],
                showBanner: true
            )
        } catch {
            errorMessage = "Unable to save risk configuration: \(error.localizedDescription)"
        }
    }

    func dismissSuccess() {
        successMessage = nil
    }

    private func riskPerTradeText(_ value: Double) -> String {
        String(format: "%.2f%%", value * 100)
    }
}

private extension ClosedRange where Bound == Double {
    func clamped(_ value: Double) -> Double {
        if value < lowerBound { return lowerBound }
        if value > upperBound { return upperBound }
        return value
    }
}
