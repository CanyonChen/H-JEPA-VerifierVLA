"""Exercise delayed heartbeat acknowledgments over a real local WebSocket."""

import asyncio
from contextlib import asynccontextmanager, suppress
import unittest
from unittest import mock

import numpy as np
from openpi_client import msgpack_numpy
from openpi.serving import websocket_policy_server as transport
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosedError

from scripts.serve_pi05_libero import RecordedPolicy, WEBSOCKET_PING_TIMEOUT


class Pi05TransportTest(unittest.IsolatedAsyncioTestCase):
    async def exercise_delayed_heartbeat(self, ping_timeout, expect_timeout=False):
        ping_seen = asyncio.Event()

        class DelayedPongConnection(ServerConnection):
            async def ping(self, data=None):
                # Delay reporting a real Pong to the heartbeat monitor. The
                # wire protocol and inference request/response remain real.
                pong = await super().ping(data)
                await pong
                ping_seen.set()
                return asyncio.create_task(asyncio.sleep(0.25, result=0.25))

        class Policy:
            def infer(self, observation):
                return {"actions": np.full((10, 7), observation["value"], dtype=np.float32)}

        provenance = {"service_id": "transport-test", "request_count": 0}
        server = transport.WebsocketPolicyServer(
            RecordedPolicy(Policy(), provenance), host="127.0.0.1", port=0,
            metadata={"evaluation": provenance}, ping_timeout=ping_timeout)
        listening = asyncio.get_running_loop().create_future()
        real_serve = transport._server.serve

        @asynccontextmanager
        async def accelerated_serve(*args, **kwargs):
            # Only accelerate the ping interval; the timeout comes from the
            # production server's run() and the project service setting.
            async with real_serve(*args, **kwargs, ping_interval=0.02,
                                  create_connection=DelayedPongConnection) as listener:
                listening.set_result(listener)
                yield listener

        with mock.patch.object(transport._server, "serve", accelerated_serve):
            server_task = asyncio.create_task(server.run())
            try:
                listener = await asyncio.wait_for(listening, timeout=5)
                port = listener.sockets[0].getsockname()[1]
                async with connect("ws://127.0.0.1:{}".format(port), ping_interval=None) as client:
                    metadata = msgpack_numpy.unpackb(await asyncio.wait_for(client.recv(), timeout=5))
                    self.assertEqual(metadata["evaluation"]["service_id"], "transport-test")
                    if not expect_timeout:
                        self.assertIsNone(next(iter(listener.connections)).ping_timeout)
                    for request_id in (1, 2):
                        await client.send(msgpack_numpy.packb({"value": request_id}))
                        response = msgpack_numpy.unpackb(await asyncio.wait_for(client.recv(), timeout=5))
                        self.assertEqual(response["evaluation"]["request_id"], request_id)
                        np.testing.assert_array_equal(response["actions"], np.full((10, 7), request_id))
                        if request_id == 1:
                            await asyncio.wait_for(ping_seen.wait(), timeout=5)
                            if expect_timeout:
                                with self.assertRaises(ConnectionClosedError) as caught:
                                    await asyncio.wait_for(client.recv(), timeout=5)
                                self.assertEqual(caught.exception.rcvd.code, 1011)
                                self.assertEqual(caught.exception.rcvd.reason, "keepalive ping timeout")
                                self.assertEqual(provenance["request_count"], 1)
                                return
                            await asyncio.sleep(0.35)
                    self.assertEqual(provenance["request_count"], 2)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_finite_heartbeat_timeout_reproduces_reported_close(self):
        await self.exercise_delayed_heartbeat(0.05, expect_timeout=True)

    async def test_project_setting_keeps_connection_and_request_sequence_after_delay(self):
        await self.exercise_delayed_heartbeat(WEBSOCKET_PING_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
