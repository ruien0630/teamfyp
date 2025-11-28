import { Request, Response, NextFunction } from 'express';

export const authenticate = (req: Request, res: Response, next: NextFunction) => {
    // Authentication logic here
    const token = req.headers['authorization'];

    if (!token) {
        return res.status(401).json({ message: 'Unauthorized' });
    }

    // Verify token logic (e.g., using JWT)
    // If valid, proceed to the next middleware
    // If invalid, return an unauthorized response

    next();
};

export const authorize = (roles: string[]) => {
    return (req: Request, res: Response, next: NextFunction) => {
        // Authorization logic here
        const userRole = req.user?.role; // Assuming req.user is set after authentication

        if (!roles.includes(userRole)) {
            return res.status(403).json({ message: 'Forbidden' });
        }

        next();
    };
};