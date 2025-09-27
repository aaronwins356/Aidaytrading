import SwiftUI

struct ChangeLogList: View {
    let entries: [AdminChangeLogEntry]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(entries) { entry in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(entry.summary)
                            .font(.subheadline.bold())
                            .foregroundColor(Theme.primaryText)
                        Spacer()
                        Text(Self.formatter.string(from: entry.timestamp))
                            .font(.caption)
                            .foregroundColor(Theme.secondaryText)
                    }
                    Text(entry.details)
                        .font(.footnote)
                        .foregroundColor(Theme.secondaryText)
                    Text("By \(entry.actor) • \(entry.categoryLabel)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Theme.cardBackground)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            if entries.isEmpty {
                Text("No recent changes. Adjust guardrails or manage users to see updates here.")
                    .font(.footnote)
                    .foregroundColor(Theme.secondaryText)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .center)
                    .background(Theme.cardBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
        }
    }

    private static let formatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }()
}

private extension AdminChangeLogEntry {
    var categoryLabel: String {
        switch category {
        case .risk: return "Risk"
        case .user: return "User"
        case .bot: return "Bot"
        }
    }
}
