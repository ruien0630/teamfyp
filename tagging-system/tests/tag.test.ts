import { TagService } from '../src/services/tagService';
import { TagRepository } from '../src/repositories/tagRepository';

describe('Tag Service', () => {
    let tagService: TagService;
    let tagRepository: TagRepository;

    beforeEach(() => {
        tagRepository = new TagRepository();
        tagService = new TagService(tagRepository);
    });

    it('should add a new tag', async () => {
        const tagData = { name: 'Test Tag', description: 'A tag for testing' };
        const savedTag = await tagService.addTag(tagData);
        expect(savedTag).toHaveProperty('id');
        expect(savedTag.name).toBe(tagData.name);
    });

    it('should fetch all tags', async () => {
        const tags = await tagService.fetchTags();
        expect(Array.isArray(tags)).toBe(true);
    });

    it('should modify an existing tag', async () => {
        const tagData = { name: 'Updated Tag', description: 'An updated tag' };
        const updatedTag = await tagService.modifyTag(1, tagData); // Assuming 1 is a valid tag ID
        expect(updatedTag.name).toBe(tagData.name);
    });

    it('should remove a tag', async () => {
        const result = await tagService.removeTag(1); // Assuming 1 is a valid tag ID
        expect(result).toBe(true);
    });
});