import SwiftUI

struct AdminView: View {
    var body: some View {
        NavigationStack {
            List {
                Section("Risk configuration") {
                    Label("Manage risk rules", systemImage: "shield.lefthalf.filled")
                }
                Section("User management") {
                    Label("Approve or disable users", systemImage: "person.crop.circle.badge.checkmark")
                }
            }
            .navigationTitle("Admin")
        }
    }
}
