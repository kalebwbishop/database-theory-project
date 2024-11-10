/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.controlprogram.federated.monitoring;

import io.netty.handler.codec.http.HttpRequest;

public class Request {
	private HttpRequest _context;
	private String _body;

	public HttpRequest getContext() {
		return _context;
	}

	public void setContext(final HttpRequest requestContext) {
		this._context = requestContext;
	}

	public String getBody() {
		return _body;
	}

	public void setBody(final String content) {
		this._body = content;
	}
}
