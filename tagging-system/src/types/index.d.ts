interface Tag {
    id: string;
    name: string;
    description?: string;
}

interface TagService {
    addTag(tag: Tag): Promise<Tag>;
    fetchTags(): Promise<Tag[]>;
    modifyTag(id: string, tag: Partial<Tag>): Promise<Tag | null>;
    removeTag(id: string): Promise<boolean>;
}

interface TagRepository {
    saveTag(tag: Tag): Promise<Tag>;
    findAllTags(): Promise<Tag[]>;
    findTagById(id: string): Promise<Tag | null>;
    deleteTag(id: string): Promise<boolean>;
}

export { Tag, TagService, TagRepository };