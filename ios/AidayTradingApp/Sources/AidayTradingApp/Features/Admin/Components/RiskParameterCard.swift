import SwiftUI

struct RiskParameterCard: View {
    let title: String
    let subtitle: String
    let value: Binding<Double>
    let range: ClosedRange<Double>
    let step: Double
    let displayValue: (Double) -> String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(Theme.primaryText)
                    Text(subtitle)
                        .font(.footnote)
                        .foregroundColor(Theme.secondaryText)
                }
                Spacer()
                Text(displayValue(value.wrappedValue))
                    .font(.title3.bold())
                    .foregroundColor(Theme.accentGreen)
            }
            Slider(value: value, in: range, step: step)
                .tint(Theme.accentGreen)
            Stepper(value: value, in: range, step: step) {
                Text("Adjust")
                    .font(.caption)
                    .foregroundColor(Theme.secondaryText)
            }
            .labelsHidden()
        }
        .padding()
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(color: Color.black.opacity(0.2), radius: 8, x: 0, y: 4)
    }
}

struct RiskIntegerCard: View {
    let title: String
    let subtitle: String
    let value: Binding<Int>
    let range: ClosedRange<Int>

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(Theme.primaryText)
                    Text(subtitle)
                        .font(.footnote)
                        .foregroundColor(Theme.secondaryText)
                }
                Spacer()
                Text("\(value.wrappedValue)")
                    .font(.title3.bold())
                    .foregroundColor(Theme.accentGreen)
            }
            Stepper(value: value, in: range) {
                Text("Max positions")
                    .font(.caption)
                    .foregroundColor(Theme.secondaryText)
            }
            .labelsHidden()
        }
        .padding()
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(color: Color.black.opacity(0.2), radius: 8, x: 0, y: 4)
    }
}
