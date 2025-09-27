import SwiftUI

struct FiltersView: View {
    enum OutcomeOption: String, CaseIterable, Identifiable {
        case all
        case win
        case loss

        var id: String { rawValue }
    }

    let symbols: [String]
    @Binding var filters: TradesViewModel.TradesFilters
    let onApply: (TradesViewModel.TradesFilters) -> Void
    let onReset: () -> Void

    @State private var outcomeSelection: OutcomeOption = .all

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Filters")
                .font(.headline)
                .foregroundColor(Theme.primaryText)
            symbolMenu
            outcomePicker
            datePickers
            HStack {
                Button("Reset") {
                    filters = .empty
                    outcomeSelection = .all
                    onReset()
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.accentRed)
                Spacer()
            }
        }
        .padding(16)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .onAppear {
            syncOutcomeSelection()
        }
    }

    private var symbolMenu: some View {
        Menu {
            Button("All Symbols") {
                filters.symbol = nil
                onApply(filters)
            }
            ForEach(symbols, id: \.self) { symbol in
                Button(symbol) {
                    filters.symbol = symbol
                    onApply(filters)
                }
            }
        } label: {
            HStack {
                Label(filters.symbol ?? "All Symbols", systemImage: "line.3.horizontal.decrease.circle")
                    .foregroundColor(Theme.primaryText)
                Spacer()
                Image(systemName: "chevron.down")
                    .foregroundColor(Theme.secondaryText)
            }
            .padding(12)
            .background(Theme.background.opacity(0.2))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
    }

    private var outcomePicker: some View {
        Picker("Outcome", selection: $outcomeSelection) {
            Text("All").tag(OutcomeOption.all)
            Text("Wins").tag(OutcomeOption.win)
            Text("Losses").tag(OutcomeOption.loss)
        }
        .pickerStyle(.segmented)
        .onChange(of: outcomeSelection) { newValue in
            switch newValue {
            case .all:
                filters.outcome = nil
            case .win:
                filters.outcome = .win
            case .loss:
                filters.outcome = .loss
            }
            onApply(filters)
        }
    }

    private var datePickers: some View {
        VStack(alignment: .leading, spacing: 8) {
            DatePicker(
                "From",
                selection: Binding(
                    get: { filters.startDate ?? Date().addingTimeInterval(-7 * 24 * 60 * 60) },
                    set: { newValue in
                        filters.startDate = newValue
                        onApply(filters)
                    }
                ),
                displayedComponents: .date
            )
            .foregroundColor(Theme.primaryText)
            HStack {
                DatePicker(
                    "To",
                    selection: Binding(
                        get: { filters.endDate ?? Date() },
                        set: { newValue in
                            filters.endDate = newValue
                            onApply(filters)
                        }
                    ),
                    displayedComponents: .date
                )
                .foregroundColor(Theme.primaryText)
                Button("Clear") {
                    filters.startDate = nil
                    filters.endDate = nil
                    onApply(filters)
                }
                .font(.footnote)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Theme.background.opacity(0.2))
                .clipShape(Capsule())
            }
        }
    }

    private func syncOutcomeSelection() {
        switch filters.outcome {
        case .none:
            outcomeSelection = .all
        case .some(.win):
            outcomeSelection = .win
        case .some(.loss):
            outcomeSelection = .loss
        }
    }
}
