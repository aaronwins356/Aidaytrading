import CoreData
import Foundation

@objc(NotificationEntity)
final class NotificationEntity: NSManagedObject {
    @NSManaged var id: String
    @NSManaged var title: String
    @NSManaged var body: String
    @NSManaged var timestamp: Date
    @NSManaged var kindRaw: String
    @NSManaged var originRaw: String
    @NSManaged var payload: Data
}

extension NotificationEntity {
    var kind: AppNotification.Kind {
        AppNotification.Kind(rawValue: kindRaw) ?? .system
    }

    var origin: AppNotification.Origin {
        AppNotification.Origin(rawValue: originRaw) ?? .realtime
    }

    func toDomain() -> AppNotification? {
        AppNotification(
            id: id,
            title: title,
            body: body,
            timestamp: timestamp,
            kind: kind,
            origin: origin,
            payload: payload
        )
    }
}

struct NotificationPersistenceController {
    static let shared = NotificationPersistenceController()

    let container: NSPersistentContainer

    init(inMemory: Bool = false) {
        container = NSPersistentContainer(name: "NotificationStore", managedObjectModel: NotificationPersistenceController.model)
        if inMemory {
            container.persistentStoreDescriptions.first?.url = URL(fileURLWithPath: "/dev/null")
        }
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
        container.loadPersistentStores { _, error in
            if let error {
                assertionFailure("Failed to load notification store: \(error)")
            }
        }
    }

    static var model: NSManagedObjectModel = {
        let model = NSManagedObjectModel()
        let entity = NSEntityDescription()
        entity.name = "NotificationEntity"
        entity.managedObjectClassName = NSStringFromClass(NotificationEntity.self)

        let idAttribute = NSAttributeDescription()
        idAttribute.name = "id"
        idAttribute.attributeType = .stringAttributeType
        idAttribute.isOptional = false

        let titleAttribute = NSAttributeDescription()
        titleAttribute.name = "title"
        titleAttribute.attributeType = .stringAttributeType
        titleAttribute.isOptional = false

        let bodyAttribute = NSAttributeDescription()
        bodyAttribute.name = "body"
        bodyAttribute.attributeType = .stringAttributeType
        bodyAttribute.isOptional = false

        let timestampAttribute = NSAttributeDescription()
        timestampAttribute.name = "timestamp"
        timestampAttribute.attributeType = .dateAttributeType
        timestampAttribute.isOptional = false

        let kindAttribute = NSAttributeDescription()
        kindAttribute.name = "kindRaw"
        kindAttribute.attributeType = .stringAttributeType
        kindAttribute.isOptional = false

        let originAttribute = NSAttributeDescription()
        originAttribute.name = "originRaw"
        originAttribute.attributeType = .stringAttributeType
        originAttribute.isOptional = false

        let payloadAttribute = NSAttributeDescription()
        payloadAttribute.name = "payload"
        payloadAttribute.attributeType = .binaryDataAttributeType
        payloadAttribute.isOptional = false

        entity.properties = [idAttribute, titleAttribute, bodyAttribute, timestampAttribute, kindAttribute, originAttribute, payloadAttribute]
        entity.uniquenessConstraints = [["id"]]

        model.entities = [entity]
        return model
    }()

    func save(notification: AppNotification) throws {
        let context = container.viewContext
        let fetch: NSFetchRequest<NotificationEntity> = NotificationEntity.fetchRequest()
        fetch.predicate = NSPredicate(format: "id == %@", notification.id)
        let existing = try context.fetch(fetch).first
        let entity = existing ?? NotificationEntity(context: context)
        entity.id = notification.id
        entity.title = notification.title
        entity.body = notification.body
        entity.timestamp = notification.timestamp
        entity.kindRaw = notification.kind.rawValue
        entity.originRaw = notification.origin.rawValue
        entity.payload = notification.payload
        try context.save()
    }

    func fetchNotifications(limit: Int? = nil) throws -> [AppNotification] {
        let request: NSFetchRequest<NotificationEntity> = NotificationEntity.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        if let limit { request.fetchLimit = limit }
        let results = try container.viewContext.fetch(request)
        return results.compactMap { $0.toDomain() }
    }

    func delete(notificationID: String) throws {
        let context = container.viewContext
        let request: NSFetchRequest<NotificationEntity> = NotificationEntity.fetchRequest()
        request.predicate = NSPredicate(format: "id == %@", notificationID)
        if let entity = try context.fetch(request).first {
            context.delete(entity)
            try context.save()
        }
    }
}

private extension NotificationEntity {
    @nonobjc static func fetchRequest() -> NSFetchRequest<NotificationEntity> {
        NSFetchRequest<NotificationEntity>(entityName: "NotificationEntity")
    }
}
