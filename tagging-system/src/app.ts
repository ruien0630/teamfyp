import express from 'express';
import { json } from 'body-parser';
import { setRoutes } from './routes/tagRoutes';
import { connectDB } from './db/index';
import { logger } from './utils/logger';
import { authMiddleware } from './middlewares/auth';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(json());
app.use(authMiddleware);

// Connect to the database
connectDB()
  .then(() => {
    logger.info('Database connected successfully');
  })
  .catch((error) => {
    logger.error('Database connection failed:', error);
  });

// Set up routes
setRoutes(app);

// Start the server
app.listen(PORT, () => {
  logger.info(`Server is running on http://localhost:${PORT}`);
});