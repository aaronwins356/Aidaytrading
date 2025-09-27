import SwiftUI

struct MetricCard: View {
    enum Tint {
        case neutral
        case positive
        case negative

        var color: Color {
            switch self {
            case .neutral: return Theme.primaryText
            case .positive: return Theme.accentGreen
            case .negative: return Theme.accentRed
            }
        }
    }

    let title: String
    let value: String
    let subtitle: String?
    let tint: Tint

    init(title: String, value: String, subtitle: String? = nil, tint: Tint = .neutral) {
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.tint = tint
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption.weight(.semibold))
                .foregroundColor(Theme.secondaryText)
            Text(value)
                .font(Theme.Typography.metric)
                .foregroundColor(tint.color)
                .minimumScaleFactor(0.6)
                .accessibilityLabel("\(title): \(value)")
            if let subtitle {
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(Theme.secondaryText)
            }
        }
        .padding(16)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}
