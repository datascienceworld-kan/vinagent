import type {IncomingWebSocketMessage, OutgoingQueryMessage} from "../types/types.ts";

class WebSocketService {

    private ws: WebSocket | null = null;
    private url: string;
    private listeners: { [event: string]: ((data: unknown) => void)[] } = {};
    private messageListeners: ((data: IncomingWebSocketMessage) => void)[] = [];

    private reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
    private reconnectAttempts: number = 0;
    private maxReconnectAttempts: number = 15;
    private initialReconnectDelay: number = 1000;

    constructor(url: string) {
        this.url = url;
    }

    connect(): void {
        if (this.ws && this.ws.readyState !== WebSocket.CLOSED) {
            console.log('WebSocket: connection is already connecting or open');
            return;
        }

        console.log('WebSocket: Attempting to connect...');
        this.emit('connecting', null);

        if (this.reconnectTimeoutId) {
            clearTimeout(this.reconnectTimeoutId);
            this.reconnectTimeoutId = null;
        }

        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('WebSocket: connection opened.');
            this.emit('open', null);

            this.reconnectAttempts = 0;
            console.log('WebSocket: Reconnection successful. Attempts reset.');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.emit('message', data);
                this.messageListeners.forEach(listener => listener(data));
            } catch (error) {
                console.error('WebSocket: error parsing message data.', error);
                this.emit('error', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket: Error:', error);
            this.emit('error', error);
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket: Closed:', event.code, event.reason);
            this.emit('close', event);
            this.ws = null;

            this.handleReconnection();
        };
    }

    private handleReconnection(): void {
        this.reconnectAttempts++;
        console.log(`WebSocket: Reconnect attempt ${this.reconnectAttempts} of ${this.maxReconnectAttempts}.`);

        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const delay = this.initialReconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            const clampedDelay = Math.min(delay, 30000);

            console.log(`WebSocket: Attempting to reconnect in ${clampedDelay}ms.`);
            this.emit('reconnecting', clampedDelay);

            this.reconnectTimeoutId = setTimeout(() => {
                this.connect();
            }, clampedDelay);

        } else {
            console.error(`WebSocket: Max reconnect attempts (${this.maxReconnectAttempts}) reached. Giving up.`);
            this.emit('error', new Error('Max reconnect attempts reached.'));
            this.emit('disconnected', null);
        }
    }

    sendMessage(message: OutgoingQueryMessage): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket: Error: Cannot send message, connection is not open.');
        }
    }

    close(): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.close();
        }
    }

    onMessage(listener: (data: IncomingWebSocketMessage) => void): () => void {
        this.messageListeners.push(listener);
        return () => {
            this.messageListeners = this.messageListeners.filter(l => l !== listener);
        };
    }

    on(event: string, listener: (data: unknown) => void): () => void {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(listener);

        return () => {
            this.listeners[event] = this.listeners[event].filter(l => l !== listener);
        };
    }

    private emit(event: string, data: unknown): void {
        if (this.listeners[event]) {
            this.listeners[event].forEach(listener => listener(data));
        }
    }

}

const websocketService = new WebSocketService('ws://localhost:8888/ws/agent');

export default websocketService;