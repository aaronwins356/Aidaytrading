import CoreData
import Foundation

@objc(AdminChangeLogEntity)
final class AdminChangeLogEntity: NSManagedObject {
    @NSManaged var id: UUID
    @NSManaged var timestamp: Date
    @NSManaged var actor: String
    @NSManaged var summary: String
    @NSManaged var details: String
    @NSManaged var categoryRaw: String
}

extension AdminChangeLogEntity {
    var category: AdminChangeLogEntry.Category {
        AdminChangeLogEntry.Category(rawValue: categoryRaw) ?? .systemFallback
    }

    func toDomain() -> AdminChangeLogEntry {
        AdminChangeLogEntry(
            id: id,
            timestamp: timestamp,
            actor: actor,
            summary: summary,
            details: details,
            category: AdminChangeLogEntry.Category(rawValue: categoryRaw) ?? .risk
        )
    }
}

private extension AdminChangeLogEntry.Category {
    static var systemFallback: AdminChangeLogEntry.Category { .risk }
}

protocol AdminChangeLogRepositoryProtocol {
    func record(_ entry: AdminChangeLogEntry) throws
    func fetchLatest(limit: Int) throws -> [AdminChangeLogEntry]
}

struct AdminChangeLogRepository: AdminChangeLogRepositoryProtocol {
    private let container: NSPersistentContainer

    init(container: NSPersistentContainer = AdminChangeLogRepository.makeContainer()) {
        self.container = container
    }

    func record(_ entry: AdminChangeLogEntry) throws {
        let context = container.viewContext
        let entity = AdminChangeLogEntity(context: context)
        entity.id = entry.id
        entity.timestamp = entry.timestamp
        entity.actor = entry.actor
        entity.summary = entry.summary
        entity.details = entry.details
        entity.categoryRaw = entry.category.rawValue
        try context.save()
    }

    func fetchLatest(limit: Int) throws -> [AdminChangeLogEntry] {
        let request: NSFetchRequest<AdminChangeLogEntity> = AdminChangeLogEntity.fetchRequest()
        request.fetchLimit = limit
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        let results = try container.viewContext.fetch(request)
        return results.map { $0.toDomain() }
    }

    private static func makeContainer() -> NSPersistentContainer {
        let container = NSPersistentContainer(name: "AdminChangeLog", managedObjectModel: Self.model)
        container.loadPersistentStores { _, error in
            if let error {
                assertionFailure("Failed to load change log store: \(error)")
            }
        }
        return container
    }

    private static var model: NSManagedObjectModel = {
        let model = NSManagedObjectModel()
        let entity = NSEntityDescription()
        entity.name = "AdminChangeLogEntity"
        entity.managedObjectClassName = NSStringFromClass(AdminChangeLogEntity.self)

        let id = NSAttributeDescription()
        id.name = "id"
        id.attributeType = .UUIDAttributeType
        id.isOptional = false

        let timestamp = NSAttributeDescription()
        timestamp.name = "timestamp"
        timestamp.attributeType = .dateAttributeType
        timestamp.isOptional = false

        let actor = NSAttributeDescription()
        actor.name = "actor"
        actor.attributeType = .stringAttributeType
        actor.isOptional = false

        let summary = NSAttributeDescription()
        summary.name = "summary"
        summary.attributeType = .stringAttributeType
        summary.isOptional = false

        let details = NSAttributeDescription()
        details.name = "details"
        details.attributeType = .stringAttributeType
        details.isOptional = false

        let category = NSAttributeDescription()
        category.name = "categoryRaw"
        category.attributeType = .stringAttributeType
        category.isOptional = false

        entity.properties = [id, timestamp, actor, summary, details, category]
        entity.uniquenessConstraints = [["id"]]
        model.entities = [entity]
        return model
    }()
}

private extension AdminChangeLogEntity {
    @nonobjc static func fetchRequest() -> NSFetchRequest<AdminChangeLogEntity> {
        NSFetchRequest<AdminChangeLogEntity>(entityName: "AdminChangeLogEntity")
    }
}
