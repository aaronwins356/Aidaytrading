import SwiftUI

struct UserRowView: View {
    let user: AdminUser

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: iconName)
                .foregroundColor(iconColor)
                .imageScale(.large)
                .frame(width: 32, height: 32)
                .background(iconColor.opacity(0.1))
                .clipShape(Circle())

            VStack(alignment: .leading, spacing: 4) {
                Text(user.username)
                    .font(.headline)
                    .foregroundColor(Theme.primaryText)
                Text(user.email)
                    .font(.footnote)
                    .foregroundColor(Theme.secondaryText)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text(user.role.displayName)
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(roleColor)
                    .clipShape(Capsule())
                HStack(spacing: 4) {
                    Image(systemName: user.status.iconName)
                    Text(user.status.displayName)
                }
                .font(.caption)
                .foregroundColor(statusColor)
            }
        }
        .padding(.vertical, 8)
    }

    private var iconName: String {
        if user.status == .pending { return "hourglass" }
        return user.role == .admin ? "lock.shield" : "person"
    }

    private var iconColor: Color {
        if user.status == .pending { return .orange }
        return user.role == .admin ? Theme.accentGreen : .blue
    }

    private var roleColor: Color {
        user.role == .admin ? Theme.accentGreen : Color.blue
    }

    private var statusColor: Color {
        switch user.status {
        case .active: return Theme.accentGreen
        case .disabled: return Theme.accentRed
        case .pending: return .orange
        }
    }
}
