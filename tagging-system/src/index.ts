import express from 'express';
import { connectToDatabase } from './db';
import { setRoutes } from './routes/tagRoutes';
import { logger } from './utils/logger';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

connectToDatabase()
    .then(() => {
        logger.info('Database connected successfully');
        setRoutes(app);
        
        app.listen(PORT, () => {
            logger.info(`Server is running on http://localhost:${PORT}`);
        });
    })
    .catch(err => {
        logger.error('Database connection failed', err);
        process.exit(1);
    });