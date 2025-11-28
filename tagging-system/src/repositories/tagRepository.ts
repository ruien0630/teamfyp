class TagRepository {
    constructor(database) {
        this.database = database;
    }

    async saveTag(tag) {
        // Logic to save a tag to the database
        return await this.database.insert('tags', tag);
    }

    async findAllTags() {
        // Logic to retrieve all tags from the database
        return await this.database.query('SELECT * FROM tags');
    }

    async findTagById(tagId) {
        // Logic to find a tag by its ID
        return await this.database.query('SELECT * FROM tags WHERE id = ?', [tagId]);
    }

    async deleteTag(tagId) {
        // Logic to delete a tag by its ID
        return await this.database.delete('tags', tagId);
    }
}

export default TagRepository;