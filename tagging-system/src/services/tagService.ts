export class TagService {
    private tags: { id: number; name: string; description: string }[] = [];
    private currentId: number = 1;

    addTag(name: string, description: string): { id: number; name: string; description: string } {
        const newTag = { id: this.currentId++, name, description };
        this.tags.push(newTag);
        return newTag;
    }

    fetchTags(): { id: number; name: string; description: string }[] {
        return this.tags;
    }

    modifyTag(id: number, name?: string, description?: string): { id: number; name: string; description: string } | null {
        const tag = this.tags.find(tag => tag.id === id);
        if (tag) {
            if (name) tag.name = name;
            if (description) tag.description = description;
            return tag;
        }
        return null;
    }

    removeTag(id: number): boolean {
        const index = this.tags.findIndex(tag => tag.id === id);
        if (index !== -1) {
            this.tags.splice(index, 1);
            return true;
        }
        return false;
    }
}