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

package org.apache.sysds.runtime.compress.colgroup.mapping;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.IMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToChar extends AMapToData {

	private static final long serialVersionUID = 6315708056775476541L;

	private final char[] _data;

	protected MapToChar(int size) {
		this(Character.MAX_VALUE, size);
	}

	public MapToChar(int unique, int size) {
		super(Math.min(unique, Character.MAX_VALUE + 1));
		_data = new char[size];
	}

	public MapToChar(int unique, char[] data) {
		super(unique);
		_data = data;
		verify();
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.CHAR;
	}

	@Override
	public int getIndex(int n) {
		return _data[n];
	}

	@Override
	public void fill(int v) {
		Arrays.fill(_data, (char) v);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.length);
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16 + 8; // object header + object reference
		size += MemoryEstimates.charArrayCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + _data.length * 2;
	}

	@Override
	public void set(int n, int v) {
		_data[n] = (char) v;
	}

	@Override
	public int setAndGet(int n, int v) {
		return _data[n] = (char) v;
	}

	@Override
	public int size() {
		return _data.length;
	}

	@Override
	public void replace(int v, int r) {
		char cv = (char) v;
		char rv = (char) r;
		for(int i = 0; i < size(); i++)
			if(_data[i] == cv)
				_data[i] = rv;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.CHAR.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_data.length);
		final int BS = 100;
		if(_data.length > BS) {
			final byte[] buff = new byte[BS * 2];
			for(int i = 0; i < _data.length;) {
				if(i + BS <= _data.length) {
					for(int o = 0; o < BS; o++) {
						IOUtilFunctions.shortToBa(_data[i++], buff, o * 2);
					}
					out.write(buff);
				}
				else {// remaining.
					for(; i < _data.length; i++)
						out.writeChar(_data[i]);
				}
			}
		}
		else {
			for(int i = 0; i < _data.length; i++)
				out.writeChar(_data[i]);
		}

	}

	protected static MapToChar readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final char[] data = new char[length];
		for(int i = 0; i < length; i++)
			data[i] = in.readChar();
		return new MapToChar(unique, data);
	}

	protected char[] getChars() {
		return _data;
	}

	@Override
	protected void preAggregateDenseToRowBy8(double[] mV, double[] preAV, int cl, int cu, int off) {
		final int h = (cu - cl) % 8;
		off += cl;
		for(int rc = cl; rc < cl + h; rc++, off++)
			preAV[getIndex(rc)] += mV[off];
		for(int rc = cl + h; rc < cu; rc += 8, off += 8)
			preAggregateDenseToRowVec8(mV, preAV, rc, off);
	}

	@Override
	protected void preAggregateDenseToRowVec8(double[] mV, double[] preAV, int rc, int off){
		preAV[getIndex(rc)] += mV[off];
		preAV[getIndex(rc + 1)] += mV[off + 1];
		preAV[getIndex(rc + 2)] += mV[off + 2];
		preAV[getIndex(rc + 3)] += mV[off + 3];
		preAV[getIndex(rc + 4)] += mV[off + 4];
		preAV[getIndex(rc + 5)] += mV[off + 5];
		preAV[getIndex(rc + 6)] += mV[off + 6];
		preAV[getIndex(rc + 7)] += mV[off + 7];
	}

	@Override
	protected void preAggregateDenseMultiRowContiguousBy8(double[] mV, int nCol, int nVal, double[] preAV, int rl,
		int ru, int cl, int cu) {
		final int h = (cu - cl) % 8;
		preAggregateDenseMultiRowContiguousBy1(mV, nCol, nVal, preAV, rl, ru, cl, cl + h);
		final int offR = nCol * rl;
		final int offE = nCol * ru;
		for(int c = cl + h; c < cu; c += 8) {
			final int id1 = _data[c], id2 = _data[c + 1], id3 = _data[c + 2], id4 = _data[c + 3], id5 = _data[c + 4],
				id6 = _data[c + 5], id7 = _data[c + 6], id8 = _data[c + 7];

			final int start = c + offR;
			final int end = c + offE;
			int nValOff = 0;
			for(int off = start; off < end; off += nCol) {
				preAV[id1 + nValOff] += mV[off];
				preAV[id2 + nValOff] += mV[off + 1];
				preAV[id3 + nValOff] += mV[off + 2];
				preAV[id4 + nValOff] += mV[off + 3];
				preAV[id5 + nValOff] += mV[off + 4];
				preAV[id6 + nValOff] += mV[off + 5];
				preAV[id7 + nValOff] += mV[off + 6];
				preAV[id8 + nValOff] += mV[off + 7];
				nValOff += nVal;
			}
		}
	}

	@Override
	public int getUpperBoundValue() {
		return Character.MAX_VALUE;
	}

	@Override
	public void copyInt(int[] d) {
		for(int i = 0; i < _data.length; i++)
			_data[i] = (char) d[i];
	}

	@Override
	public void copyBit(BitSet d) {
		for(int i = d.nextSetBit(0); i >= 0; i = d.nextSetBit(i + 1)) {
			_data[i] = 1;
		}
	}

	@Override
	public int[] getCounts(int[] ret) {
		for(int i = 0; i < _data.length; i++)
			ret[_data[i]]++;
		return ret;
	}

	@Override
	public AMapToData resize(int unique) {
		final int size = _data.length;
		AMapToData ret;
		if(unique <= 1)
			return new MapToZero(size);
		else if(unique == 2 && size > 32)
			ret = new MapToBit(unique, size);
		else if(unique <= 127)
			ret = new MapToUByte(unique, size);
		else if(unique < 256)
			ret = new MapToByte(unique, size);
		else {
			setUnique(unique);
			return this;
		}
		ret.copy(this);
		return ret;
	}

	@Override
	public int countRuns() {
		int c = 1;
		char prev = _data[0];
		for(int i = 1; i < _data.length; i++) {
			c += prev == _data[i] ? 0 : 1;
			prev = _data[i];
		}
		return c;
	}

	@Override
	public AMapToData slice(int l, int u) {
		return new MapToChar(getUnique(), Arrays.copyOfRange(_data, l, u));
	}

	@Override
	public AMapToData append(AMapToData t) {
		if(t instanceof MapToChar) {
			MapToChar tb = (MapToChar) t;
			char[] tbb = tb._data;
			final int newSize = _data.length + t.size();
			final int newDistinct = Math.max(getUnique(), t.getUnique());

			// copy
			char[] ret = Arrays.copyOf(_data, newSize);
			System.arraycopy(tbb, 0, ret, _data.length, t.size());

			return new MapToChar(newDistinct, ret);
		}
		else {
			throw new NotImplementedException("Not implemented append on Bit map different type");
		}
	}

	@Override
	public AMapToData appendN(IMapToDataGroup[] d) {
		int p = 0; // pointer
		for(IMapToDataGroup gd : d)
			p += gd.getMapToData().size();
		final char[] ret = new char[p];

		p = 0;
		for(int i = 0; i < d.length; i++) {
			if(d[i].getMapToData().size() > 0) {
				final MapToChar mm = (MapToChar) d[i].getMapToData();
				final int ms = mm.size();
				System.arraycopy(mm._data, 0, ret, p, ms);
				p += ms;
			}
		}

		return new MapToChar(getUnique(), ret);
	}

	@Override
	public int getMaxPossible() {
		return Character.MAX_VALUE;
	}

	@Override
	public boolean equals(AMapToData e) {
		return e instanceof MapToChar && //
			e.getUnique() == getUnique() && //
			Arrays.equals(((MapToChar) e)._data, _data);
	}

	@Override
	protected void preAggregateDDC_DDCSingleCol_vec(AMapToData tm, double[] td, double[] v, int r) {
		if(tm instanceof MapToChar)
			preAggregateDDC_DDCSingleCol_vecChar((MapToChar) tm, td, v, r);
		else
			super.preAggregateDDC_DDCSingleCol_vec(tm, td, v, r);
	}

	protected final void preAggregateDDC_DDCSingleCol_vecChar(MapToChar tm, double[] td, double[] v, int r) {
		final int r2 = r + 1, r3 = r + 2, r4 = r + 3, r5 = r + 4, r6 = r + 5, r7 = r + 6, r8 = r + 7;
		v[getIndex(r)] += td[tm.getIndex(r)];
		v[getIndex(r2)] += td[tm.getIndex(r2)];
		v[getIndex(r3)] += td[tm.getIndex(r3)];
		v[getIndex(r4)] += td[tm.getIndex(r4)];
		v[getIndex(r5)] += td[tm.getIndex(r5)];
		v[getIndex(r6)] += td[tm.getIndex(r6)];
		v[getIndex(r7)] += td[tm.getIndex(r7)];
		v[getIndex(r8)] += td[tm.getIndex(r8)];
	}

}
