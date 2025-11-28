class TagController {
    constructor(private tagService: TagService) {}

    async createTag(req: Request, res: Response) {
        try {
            const tagData = req.body;
            const newTag = await this.tagService.addTag(tagData);
            res.status(201).json(newTag);
        } catch (error) {
            res.status(500).json({ message: 'Error creating tag', error });
        }
    }

    async getTags(req: Request, res: Response) {
        try {
            const tags = await this.tagService.fetchTags();
            res.status(200).json(tags);
        } catch (error) {
            res.status(500).json({ message: 'Error fetching tags', error });
        }
    }

    async updateTag(req: Request, res: Response) {
        try {
            const tagId = req.params.id;
            const tagData = req.body;
            const updatedTag = await this.tagService.modifyTag(tagId, tagData);
            res.status(200).json(updatedTag);
        } catch (error) {
            res.status(500).json({ message: 'Error updating tag', error });
        }
    }

    async deleteTag(req: Request, res: Response) {
        try {
            const tagId = req.params.id;
            await this.tagService.removeTag(tagId);
            res.status(204).send();
        } catch (error) {
            res.status(500).json({ message: 'Error deleting tag', error });
        }
    }
}

export default TagController;