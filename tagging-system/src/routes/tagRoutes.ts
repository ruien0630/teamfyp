import { Router } from 'express';
import TagController from '../controllers/tagController';

const router = Router();
const tagController = new TagController();

router.post('/tags', tagController.createTag.bind(tagController));
router.get('/tags', tagController.getTags.bind(tagController));
router.put('/tags/:id', tagController.updateTag.bind(tagController));
router.delete('/tags/:id', tagController.deleteTag.bind(tagController));

export default function setRoutes(app) {
    app.use('/api', router);
}