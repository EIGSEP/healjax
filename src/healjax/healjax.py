"""
Copyright (c) 2023 ghcollin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import jax
import jax.numpy as jnp
import numpy
from functools import partial

INT_TYPE = jnp.int32
FLOAT_TYPE = jnp.float32


def atan2_2pi(y, x):
    val = jnp.arctan2(y, x)
    #return val + (val < 0)*2*jnp.pi
    return val + jnp.where(val < 0, 2*jnp.pi, 0.0)

def xyz_to_hp_polar(out_dtype, nside, vx, vy, vz_raw):
    
    phi = atan2_2pi(vy, vx)
    phi_t = jnp.fmod(phi, jnp.pi/2)
    #jax.debug.print("hj vx={}, vy={}, vz={}, phi={}, phi_t={}", vx, vy, vz_raw, phi, phi_t)

    north, vz = (vz_raw > 2.0/3.0), jnp.abs(vz_raw)

    coz = jnp.hypot(vx, vy)

    kx = (coz/jnp.sqrt(1.0 + vz)) * jnp.sqrt(3) * jnp.fabs(nside * (2.0*phi_t - jnp.pi)/jnp.pi)

    ky = (coz/jnp.sqrt(1.0 + vz)) * jnp.sqrt(3) * nside * 2.0 * phi_t / jnp.pi

    xx, yy = jax.lax.cond(north, lambda: (nside - kx, nside - ky), lambda: (ky, kx))

    x = jnp.minimum(nside - 1, jnp.floor(xx).astype(out_dtype))
    y = jnp.minimum(nside - 1, jnp.floor(yy).astype(out_dtype))

    #dx = xx - x
    #dy = yy - y

    sector = (phi - phi_t)*2/jnp.pi
    offset_raw = jnp.round(sector).astype(out_dtype)
    offset = jnp.fmod(jnp.fmod(offset_raw, 4) + 4, 4) # c '%' operators has same sematics as fmod. fmod should return int (despite name)
    column = offset
    return jnp.where(north, 0, 8) + column, x, y

def xyz_to_hp_equator(out_dtype, nside, vx, vy, vz):

    phi = atan2_2pi(vy, vx)
    phi_t = jnp.fmod(phi, jnp.pi/2.0)

    # project into the unit square z=[-2/3, 2/3], phi=[0, pi/2]
    #zunits = (vz + 2.0/3.0) * 3.0/4.0
    #zunits = (vz * 3.0/4.0 + 0.5)
    zunits = (vz + (2.0 / 3.0)) / (4.0 / 3.0)
    phiunits = phi_t * 2/jnp.pi
    # convert into diagonal units
    # (add 1 to u2 so that they both cover the range [0,2])
    u1 = zunits + phiunits
    u2 = zunits - phiunits + 1.0
    # 1.0 - phiunits = 1 - phi_t * 2/jnp.pi = (pi/2 - phi_t) 2/pi = (pi/2 - jnp.fmod(atan2_2pi(vy, vx), jnp.pi/2)) 2/pi
    #       = (pi/2 - jnp.fmod(atan2(vy, vx), jnp.pi/2)) 2/pi = (jnp.fmod(pi/2 - atan2(vy, vx), jnp.pi/2)) 2/pi
    #       = (jnp.fmod(atan2(vx, vy), jnp.pi/2)) 2/pi
    #u2 = zunits + jnp.fmod(jnp.arctan2(vx, vy), jnp.pi/2) * 2/jnp.pi
    # x is the northeast direction, y is the northwest.
    xx_full = u1 * nside
    yy_full = u2 * nside

    # 1.0 - phiunits = 1 - phi_t * 2/jnp.pi = (pi/2 - phi_t) 2/pi = (pi/2 - jnp.fmod(atan2_2pi(vy, vx), jnp.pi/2)) 2/pi = (pi/2 - atan(abs(vy/vx))) 2/pi = (acot(abs(vy/vx))) 2/pi
    # yy_full = nside ((3 vz/4 + 1.5) pi/2 - jnp.fmod(atan2_2pi(vy, vx), jnp.pi/2) ) 2/pi

    # now compute which big healpix it's in.
    # (note that we subtract off the modded portion used to
    # compute the position within the healpix, so this should be
    # very close to one of the boundaries.)
    sector = (phi - phi_t)*2/jnp.pi
    offset_raw = jnp.round(sector).astype(out_dtype)
    offset = jnp.fmod(jnp.fmod(offset_raw, 4) + 4, 4) # c '%' operators has same sematics as fmod. fmod should return int (despite name)
    
    # we're looking at a square in z,phi space with an X dividing it.
    # we want to know which section we're in.
    # xx ranges from 0 in the bottom-left to 2Nside in the top-right.
    # yy ranges from 0 in the bottom-right to 2Nside in the top-left.
    # (of the phi,z unit box)

    upper_xx = xx_full >= nside
    upper_yy = yy_full >= nside
    xx, yy = jnp.where(upper_xx, xx_full - nside, xx_full), jnp.where(upper_yy, yy_full - nside, yy_full)
    basehp = jnp.array([
        [ 8 + offset                    , offset + 4    ],
        [ jnp.fmod(offset + 1, 4) + 4   , offset        ]
    ])[upper_xx.astype(int)][upper_yy.astype(int)]

    x = jnp.maximum(0, jnp.minimum(nside-1, jnp.floor(xx).astype(out_dtype)))
    #dx = xx - x 
    y = jnp.maximum(0, jnp.minimum(nside-1, jnp.floor(yy).astype(out_dtype)))
    #dy = yy - y
    #jax.debug.print("{} {} {} {} {} {} {} {} {} {} {}", vx, vy, vz, phi, phi_t, zunits, phiunits, u2, basehp, x, y)

    return basehp, x, y

def xyz_to_hp(nside, vx, vy, vz, out_dtype=None):
    out_dtype = int if out_dtype is None else out_dtype
    return jax.lax.cond(jnp.abs(vz) >= 2.0/3.0, 
                        partial(xyz_to_hp_polar, out_dtype), 
                        partial(xyz_to_hp_equator, out_dtype), 
                        nside, vx, vy, vz)

def hp_to_zphi_polar(nside, bighp, x, y):

    zfactor, x, y = jax.lax.cond(jnp.logical_and(issouthpolar(bighp), x + y < nside),
        lambda: (-1.0, nside - y, nside - x),
        lambda: (1.0, x, y)
        )
    
    phi_t_flag = jnp.logical_or(y != nside, x != nside)
    phi_t = phi_t_flag * (jnp.pi * (nside-y) / (2.0 * ((nside-x) + (nside-y))))

    vv = jnp.where(phi_t < jnp.pi/4,
        jnp.fabs(jnp.pi * (nside - x) / ((2.0 * phi_t - jnp.pi) * nside) / jnp.sqrt(3.0)),
        jnp.fabs(jnp.pi * (nside - y) / (2.0 * phi_t * nside) / jnp.sqrt(3.0))              
    )

    z = (1 - vv) * (1 + vv)
    rad = jnp.sqrt(1.0 + z) * vv

    z = z * zfactor

    # // The big healpix determines the phi offset
    phi = jnp.where(issouthpolar(bighp),
        jnp.pi/2.0 * (bighp-8) + phi_t,
        jnp.pi/2.0 * bighp + phi_t
    )
    phi = jnp.mod(phi, 2*jnp.pi)
    return z, phi, rad

def hp_to_zphi_equator(nside, bighp, x, y):
    x = x / nside
    y = y / nside

    bighp, zoff, phioff = jax.lax.cond(
        bighp <= 3,
        lambda: (bighp, 0.0, 1.0), # // north
        lambda: jax.lax.cond(
            bighp <= 7,
            lambda: (bighp - 4, -1.0, 0.0), # // equator
            lambda: (bighp - 8, -2.0, 1.0) # // south
        )
    )

    z = 2.0/3.0 * (x + y + zoff)
    phi = jnp.pi/4 * (x - y + phioff + 2 * bighp)
    phi = jnp.mod(phi, 2*jnp.pi)
    rad = jnp.sqrt(jnp.maximum(0.0, 1 - jnp.square(z))) # This sqrt can cause spirious NaN errors in debug mode, so we clip the input
    return z, phi, rad

def hp_to_zphi(nside, bighp, xp, yp, dx, dy):

    # // this is x,y position in the healpix reference frame
    x = xp + dx
    y = yp + dy

    polar_routine = jnp.logical_or(
        jnp.logical_and(isnorthpolar(bighp), x + y > nside),
        jnp.logical_and(issouthpolar(bighp), x + y < nside)
    )

    return jax.lax.cond(polar_routine, hp_to_zphi_polar, hp_to_zphi_equator, nside, bighp, x, y)

def zphi2radec(z, phi, rad):
    d = jnp.where(jnp.fabs(z) > 0.9,
        jnp.pi/2 - jnp.arctan2(rad, z),
        jnp.arcsin(z)               
    )
    return phi, d

def zphi2xyz(z, phi, rad):
    x = rad * jnp.cos(phi)
    y = rad * jnp.sin(phi)
    return x, y, z

def xyz2radec(x, y, z):
    ra = jnp.mod(jnp.arctan2(y, x), 2 * jnp.pi)
    dec = jnp.where(jnp.fabs(z) > 0.9,
        jnp.pi/2 - jnp.arctan2(jnp.hypot(x, y), z),
        jnp.arcsin(z)               
    )
    return ra, dec

def radec2x(r, d):
    return jnp.cos(d)*jnp.cos(r)

def radec2y(r, d):
    return jnp.cos(d)*jnp.sin(r)

def radec2z(r, d):
    return jnp.sin(d)

def radec2xyz(r, d):
    return radec2x(r, d), radec2y(r, d), radec2z(r, d)

#def radec_to_healpix(nside, ra, dec):
#    return healpixl_compose_xy(nside, *xyz_to_hp(nside, *radec2xyz(ra, dec)))

def healpixl_xy_to_composed_xy(nside, hpxy, x, y):
    return ((nside * hpxy) + x) * nside + y

def healpixl_xy_to_nested(nside, hpxy, x, y):
    # We construct the index called p_n' in the healpix paper, whose bits
    # are taken from the bits of x and y:
    #    x = ... b4 b2 b0
    #    y = ... b5 b3 b1
    # We go through the bits of x,y, building up "index":

    def loop_body(i, carry):
        index, xc, yc = carry
        new_index = jnp.bitwise_or(index, jnp.left_shift(jnp.bitwise_or(jnp.left_shift(jnp.bitwise_and(yc, 1), 1), jnp.bitwise_and(xc, 1)), 2*i))
        new_y = jnp.right_shift(yc, 1)
        new_x = jnp.right_shift(xc, 1)
        #update_index = jax.lax.cond(jnp.logical_or(new_x != 0, new_y != 0), )
        return (new_index, new_x, new_y)
    
    final_index, _, _ = jax.lax.fori_loop(0, 8*x.dtype.itemsize//2, loop_body, (0, x, y))

    return final_index + hpxy.astype(int) * (nside * nside)

def healpixl_xy_to_ring(nside, hpxy, x, y):
    frow = hpxy // 4
    F1 = frow + 2
    v = x + y

    # "ring" starts from 1 at the north pole and goes to 4Nside-1 at
    # the south pole; the pixels in each ring have the same latitude.
    ring = F1*nside - v - 1

    def north_pole(ring):
        # north polar.
        # left-to-right coordinate within this healpix
        index0 = (nside - 1 - y)
        # offset from the other big healpixes
        index1 = index0 + jnp.fmod(hpxy, 4) * ring
        # offset from the other rings
        index2 = index1+  ring*(ring-1)*2
        return index2
    
    def south_pole(ring):
        # south polar.
        # Here I first flip everything so that we label the pixels
        # at zero starting in the southeast corner, increasing to the
        # west and north, then subtract that from the total number of
        # healpixels.
        ri = 4*nside - ring
        # index within this healpix
        index0 = (ri-1) - x
        # big healpixes
        index1 = index0 + ((3-jnp.fmod(hpxy, 4)) * ri)
        # other rings
        index2 = index1 + ri*(ri-1)*2
        # flip!
        index3 = 12*nside*nside - 1 - index2
        return index3
    
    def equatorial(ring):
        # equatorial.
        s = jnp.fmod((ring - nside), 2)
        F2 = 2 * jnp.fmod(hpxy, 4) - jnp.fmod(frow, 2) + 1
        h = x - y

        index0 = ((F2 * nside + h + s) / 2).astype(h.dtype)
        # offset from the north polar region:
        index1 = index0 + nside * (nside - 1) * 2
        # offset within the equatorial region:
        index2 = index1 + nside * 4 * (ring - nside)
        # handle healpix #4 wrap-around
        index3 = index2 + jnp.where(jnp.logical_and(hpxy == 4, y > x), (4 * nside - 1), 0)
        #jax.debug.print("hj {} {} {} frow={}, F1={}, v={}, ringind={}, s={}, F2={}, h={}, longind={}.", hpxy, x, y, frow, F1, v, ring, s, F2, h, index0)
        return index3
    
    return jnp.piecewise(ring, [ring <= nside, ring >= 3*nside], [north_pole, south_pole, equatorial])

def healpixl_composed_xy_to_xy(nside, hp):
    ns2 = nside * nside
    bighp = (hp/ns2).astype(hp.dtype)
    smallhp = jnp.fmod(hp, ns2)
    x = (smallhp/nside).astype(hp.dtype)
    y = jnp.fmod(smallhp, nside)
    return bighp, x, y

def healpixl_nested_to_xy(nside, hp):
    ns2 = nside * nside
    bighp = (hp/ns2).astype(hp.dtype)

    def loop_body(i, carry):
        index0, xc, yc = carry
        new_x = jnp.bitwise_or(xc, jnp.left_shift(jnp.bitwise_and(index0, 0x1), i))
        index1 = jnp.right_shift(index0, 1)
        new_y = jnp.bitwise_or(yc, jnp.left_shift(jnp.bitwise_and(index1, 0x1), i))
        index2 = jnp.right_shift(index1, 1)
        return index2, new_x, new_y

    index = jnp.fmod(hp, ns2)
    x_dtype = hp.dtype
    _, x, y = jax.lax.fori_loop(0, 8*x_dtype.itemsize//2, loop_body, (index, jnp.array(0).astype(x_dtype), jnp.array(0).astype(x_dtype)))

    return bighp, x, y

def healpixl_decompose_ring(nside, hp):
    ns2 = nside * nside
    def smallhp(): # hp < 2 * ns2
        ring = (0.5 + jnp.sqrt(0.25 + 0.5 * hp)).astype(hp.dtype)
        preoffset = 2 * ring * (ring  - 1)
        # // The sqrt above can introduce precision issues that can cause ring to
        # // be off by 1, so we check whether the offset is now larger than the HEALPix
        # // value, and if so we need to adjust ring and offset accordingly
        ring = jnp.where(preoffset > hp, ring - 1, ring)
        offset = 2 * ring * (ring  - 1)
        longind = hp - offset
        return ring, longind
    def midhp(): # hp < 10 * ns2
        preoffset = 2 * nside * (nside - 1)
        ring = ((hp - preoffset) / (nside * 4) + nside).astype(hp.dtype)
        offset = preoffset + 4 * (ring - nside) * nside
        longind = hp - offset
        return ring, longind
    def largehp(): #otherwise
        preoffset = 2 * nside * (nside - 1) + 8 * ns2
        ring = (
            (2 * nside + 1 - jnp.sqrt((2 * nside + 1) * (2 * nside + 1) - 2 * (hp - preoffset))) * 0.5
        ).astype(hp.dtype)
        offset = preoffset + 2 * ring * (2 * nside + 1 - ring)
        # // The sqrt above can introduce precision issues that can cause ring to
        # // be off by 1, so we check whether the offset is now larger than the HEALPix
        # // value, and if so we need to adjust ring and offset accordingly
        ring, postoffset = jax.lax.cond(offset > hp,
            lambda: (ring - 1, preoffset - (4 * nside - 4 * (ring - 1))),
            lambda: (ring, offset)
        )
        longind = hp - postoffset
        ring = ring + 3 * nside
        return ring, longind
    return jax.lax.cond(hp < 2 * ns2, smallhp, lambda: jax.lax.cond(hp < 10 * ns2, midhp, largehp))

def healpixl_ring_to_xy(nside, hp):
    #ns2 = nside * nside
    ringind, longind = healpixl_decompose_ring(nside, hp)
    def smallring(): # ringind <= Nside
        bighp = (longind / ringind).astype(hp.dtype)
        ind = longind - bighp * ringind
        y = (nside - 1) - ind
        frow = (bighp / 4).astype(hp.dtype)
        F1 = frow + 2
        v = F1*nside - ringind - 1
        x = v - y
        return bighp, x, y
    
    def midring(): # ringind < 3*nside
        panel = (longind / nside).astype(hp.dtype)
        ind = jnp.fmod(longind, nside)
        bottomleft = ind < ((ringind - nside + 1) / 2).astype(hp.dtype)
        topleft = ind < ((3*nside - ringind + 1) / 2).astype(hp.dtype)

        bl_tl   = [4 + panel, 0, 0]
        bl_ntl  = [8 + panel, 0, 0]
        nbl_tl  = [panel, 0, 0]
        pre_nbl_ntl = 4 + jnp.fmod(panel + 1, 4)
        nbl_ntl = jax.lax.cond(pre_nbl_ntl == 4,
            # // Gah!  Wacky hack - it seems that since
            # // "longind" is negative in this case, the
            # // rounding behaves differently, so we end up
            # // computing the wrong "h" and have to correct
            # // for it.
            lambda: [pre_nbl_ntl, (4*nside - 1), 1],
            lambda: [pre_nbl_ntl, 0, 0]
        )

        bighp_block = jnp.array([[nbl_ntl, nbl_tl], [bl_ntl, bl_tl]])
        bighp, longind_off, R = bighp_block[bottomleft.astype(int), topleft.astype(int)]
        postlongind = longind - longind_off

        frow = (bighp / 4).astype(hp.dtype)
        F1 = frow + 2
        F2 = 2 * jnp.fmod(bighp, 4) - jnp.fmod(frow, 2) + 1
        s = jnp.fmod(ringind - nside, 2)
        v = F1 * nside - ringind - 1
        h = 2 * postlongind - s - F2*nside
        h = h - R
        prex = ((v + h) / 2).astype(hp.dtype)
        prey = ((v - h) / 2).astype(hp.dtype)

        tweak = jnp.logical_or((v != (prex+prey)), (h != (prex-prey)))
        h = h + tweak
        x = ((v + h) / 2).astype(hp.dtype)
        y = ((v - h) / 2).astype(hp.dtype)
        return bighp, x, y
    
    def largering(): # otherwise
        ri = 4 * nside - ringind
        bighp = 8 + (longind / ri).astype(hp.dtype)
        ind = longind - jnp.fmod(bighp, 4) * ri
        y = (ri - 1) - ind
        frow = (bighp / 4).astype(hp.dtype)
        F1 = frow + 2
        v = F1 * nside - ringind - 1
        x = v - y
        return bighp, x, y
    return jax.lax.cond(ringind <= nside, smallring, lambda: jax.lax.cond(ringind < 3*nside, midring, largering))

to_scheme_funcs = {
    'xy': healpixl_xy_to_composed_xy,
    'nest': healpixl_xy_to_nested,
    'ring': healpixl_xy_to_ring
}

from_scheme_funcs = {
    'xy': healpixl_composed_xy_to_xy,
    'nest': healpixl_nested_to_xy,
    'ring': healpixl_ring_to_xy
}


def isnorthpolar(bighealpix):
    return bighealpix <= 3

def issouthpolar(bighealpix):
    return bighealpix >= 8

def bighealpix_get_patch(hp): # hp is bighp in this function
    #   check                       north pole          south pole          equa
    # ((dx ==  1) && (dy ==  0))    (hp + 1) % 4        4 + ((hp + 1) % 4)  hp - 4
    # ((dx ==  0) && (dy ==  1))    (hp + 3) % 4        hp - 4              (hp + 3) % 4
    # ((dx ==  1) && (dy ==  1))    (hp + 2) % 4        hp - 8              -1
    # ((dx == -1) && (dy ==  0))    (hp + 4)            8 + ((hp + 3) % 4)  8 + ((hp + 3) % 4)
    # ((dx ==  0) && (dy == -1))    4 + ((hp + 1) % 4)  8 + ((hp + 1) % 4)  hp + 4
    # ((dx == -1) && (dy == -1))    hp + 8              8 + ((hp + 2) % 4)  -1
    # ((dx ==  1) && (dy == -1))    -1                  -1                  4 + ((hp + 1) % 4)
    # ((dx == -1) && (dy ==  1))    -1                  -1                  4 + ((hp - 1) % 4)

    # rearrange this into array order, dx first, insert identity at 0, 0
    # ((dx == -1) && (dy == -1))    hp + 8              8 + ((hp + 2) % 4)  -1
    # ((dx == -1) && (dy ==  0))    (hp + 4)            8 + ((hp + 3) % 4)  8 + ((hp + 3) % 4)
    # ((dx == -1) && (dy ==  1))    -1                  -1                  4 + ((hp - 1) % 4)
    # ((dx ==  0) && (dy == -1))    4 + ((hp + 1) % 4)  8 + ((hp + 1) % 4)  hp + 4
    # dx = 0, dy = 0                hp                  hp                  hp
    # ((dx ==  0) && (dy ==  1))    (hp + 3) % 4        hp - 4              (hp + 3) % 4
    # ((dx ==  1) && (dy == -1))    -1                  -1                  4 + ((hp + 1) % 4)
    # ((dx ==  1) && (dy ==  0))    (hp + 1) % 4        4 + ((hp + 1) % 4)  hp - 4
    # ((dx ==  1) && (dy ==  1))    (hp + 2) % 4        hp - 8              -1

    north = jnp.array([
        hp + 8,
        (hp + 4),
        -1,
        4 + jnp.fmod((hp + 1), 4),
        hp,
        jnp.fmod((hp + 3), 4),
        -1,
        jnp.fmod((hp + 1), 4),
        jnp.fmod((hp + 2), 4)
    ]).reshape((3, 3))

    south = jnp.array([
        8 + jnp.fmod((hp + 2), 4),
        8 + jnp.fmod((hp + 3), 4),
        -1,
        8 + jnp.fmod((hp + 1), 4),
        hp,
        hp - 4,
        -1,
        4 + jnp.fmod((hp + 1), 4),
        hp - 8  
    ]).reshape((3, 3))

    equa = jnp.array([
        -1,
        8 + jnp.fmod((hp + 3), 4),
        4 + jnp.fmod((hp - 1), 4),
        hp + 4,
        hp,
        jnp.fmod((hp + 3), 4),
        4 + jnp.fmod((hp + 1), 4),
        hp - 4,
        -1
    ]).reshape((3, 3))

    return jnp.where(isnorthpolar(hp), north, jnp.where(issouthpolar(hp), south, equa))
    
def healpix_get_patch_xy(nside, bighp, x, y):
    x_offsets, y_offsets = numpy.meshgrid([-1, 0, 1], [-1, 0, 1])#, indexing='ij')

    #bighp_x_offsets = jnp.where(jnp.logical_and(x == 0, x == nside-1), x_offsets, jnp.zeros_like(x_offsets))
    #bighp_y_offsets = jnp.where(jnp.logical_and(y == 0, y == nside-1), y_offsets, jnp.zeros_like(y_offsets))

    neighbour_x_vals = jnp.fmod(x + nside + x_offsets, nside)
    bighp_x_offsets = jnp.floor((x + x_offsets)/nside).astype(x.dtype)
    neighbour_y_vals = jnp.fmod(y + nside + y_offsets, nside)
    bighp_y_offsets = jnp.floor((y + y_offsets)/nside).astype(y.dtype)

    def north(xval, bighp_x_off, yval, bighp_y_off):
        xalter = bighp_x_off == 1
        xpreswap = jnp.where(xalter, nside - 1, xval)
        yalter = bighp_y_off == 1
        ypreswap = jnp.where(yalter, nside - 1, yval)
        xfinal, yfinal = jax.lax.cond(jnp.logical_or(xalter, yalter), lambda: (ypreswap, xpreswap), lambda: (xpreswap, ypreswap))
        return xfinal, yfinal
    
    def south(xval, bighp_x_off, yval, bighp_y_off):
        xalter = bighp_x_off == -1
        xpreswap = jnp.where(xalter, 0, xval)
        yalter = bighp_y_off == -1
        ypreswap = jnp.where(yalter, 0, yval)
        xfinal, yfinal = jax.lax.cond(jnp.logical_or(xalter, yalter), lambda: (ypreswap, xpreswap), lambda: (xpreswap, ypreswap))
        return xfinal, yfinal

    final_x_vals, final_y_vals = jax.lax.cond(isnorthpolar(bighp), jax.vmap(jax.vmap(north)), partial(jax.lax.cond, issouthpolar(bighp), jax.vmap(jax.vmap(south)), lambda xv, _, yv, __: (xv, yv)), neighbour_x_vals, bighp_x_offsets, neighbour_y_vals, bighp_y_offsets)

    bighp_patch = bighealpix_get_patch(bighp)

    final_hps = jax.vmap(jax.vmap(lambda x_off, y_off: bighp_patch[x_off+1, y_off+1]))(bighp_x_offsets, bighp_y_offsets)

    return final_hps, final_x_vals, final_y_vals
    
def healpixl_get_neighbours_xy(nside, bighp, x, y):
    # 0 : +, 0 -> 2, 1
    # 1 : +, + -> 2, 2
    # 2 : 0, + -> 1, 2
    # 3 : -, + -> 0, 2,
    # 4 : -, 0 -> 0, 1,
    # 5 : -, - -> 0, 0
    # 6 : 0, - -> 1, 0
    # 7 : +, - -> 2, 0

    # neighbour_coords = numpy.array([
    #     [2, 1],
    #     [2, 2],
    #     [1, 2],
    #     [0, 2],
    #     [0, 1],
    #     [0, 0],
    #     [1, 0],
    #     [2, 0]
    # ])

    neighbour_coords = numpy.array([
        [1, 0],
        [2, 0],
        [2, 1],
        [2, 2],
        [1, 2],
        [0, 2],
        [0, 1],
        [0, 0]
    ])

    # neighbour_coords = numpy.array([
    #     [0, 1],
    #     [0, 0],
    #     [1, 0],
    #     [2, 0],
    #     [2, 1],
    #     [2, 2],
    #     [1, 2],
    #     [0, 2]
    # ])

    neighbour_coords_i = neighbour_coords[:, 0] # [ neighbour_coords[i][0] for i in range(len(neighbour_coords)) ]
    neighbour_coords_j = neighbour_coords[:, 1] # [ neighbour_coords[i][1] for i in range(len(neighbour_coords)) ]

    patch_hps, patch_xs, patch_ys = healpix_get_patch_xy(nside, bighp, x, y)
    return patch_hps[neighbour_coords_i, neighbour_coords_j], patch_xs[neighbour_coords_i, neighbour_coords_j], patch_ys[neighbour_coords_i, neighbour_coords_j]

def capture_m1s(f):
    def wrap(bighp, x, y):
        return jnp.where(bighp < 0,
            -1,
            f(bighp, x, y)
        )
    return wrap

##############
# API funcs
##############

def bighpxy2scheme(scheme, nside, bighp, x, y):
    return to_scheme_funcs[scheme](nside, bighp, x, y)

def scheme2bighpxy(scheme, nside, hp_idx):
    return from_scheme_funcs[scheme](nside, hp_idx)

def ang2vec_radec(ra, dec):
    return radec2xyz(ra, dec)

def ang2vec(theta, phi):
    ra = phi
    dec = jnp.pi/2 - theta
    return ang2vec_radec(ra, dec)

def vec2pix(scheme, nside, x, y, z, out_dtype=None):
    return to_scheme_funcs[scheme](nside, *xyz_to_hp(nside, x, y, z, out_dtype=out_dtype))

def ang2pix_radec(scheme, nside, ra, dec, out_dtype=None):
    return vec2pix(scheme, nside, *ang2vec_radec(ra, dec), out_dtype=out_dtype)

def ang2pix(scheme, nside, theta, phi, out_dtype=None):
    return vec2pix(scheme, nside, *ang2vec(theta, phi), out_dtype=out_dtype)

def vec2ang_radec(x, y, z):
    return xyz2radec(x, y, z)
    
def vec2ang(x, y, z):
    ra, dec = vec2ang_radec(x, y, z)
    phi = ra
    theta = jnp.pi/2 - dec
    return theta,  phi

def pix2vec(scheme, nside, hp, dx=None, dy=None):
    dx = 0.5 if dx is None else dx
    dy = 0.5 if dy is None else dy
    return zphi2xyz(*hp_to_zphi(nside, *from_scheme_funcs[scheme](nside, hp), dx, dy))

def pix2ang_radec(scheme, nside, hp, dx=None, dy=None):
    dx = 0.5 if dx is None else dx
    dy = 0.5 if dy is None else dy
    return zphi2radec(*hp_to_zphi(nside, *from_scheme_funcs[scheme](nside, hp), dx, dy))

def pix2ang_colatlong(scheme, nside, hp, dx=None, dy=None):
    dx = 0.5 if dx is None else dx
    dy = 0.5 if dy is None else dy
    ra, dec = zphi2radec(*hp_to_zphi(nside, *from_scheme_funcs[scheme](nside, hp), dx, dy))
    phi = ra
    theta = jnp.pi/2 - dec
    return theta, phi

def pix2ang(scheme, nside, hp, dx=None, dy=None):
    dx = 0.5 if dx is None else dx
    dy = 0.5 if dy is None else dy
    return vec2ang(*zphi2xyz(*hp_to_zphi(nside, *from_scheme_funcs[scheme](nside, hp), dx, dy)))

def get_patch(scheme, nside, hp):
    return jax.vmap(jax.vmap(capture_m1s(partial(to_scheme_funcs[scheme], nside))))(*healpix_get_patch_xy(nside, *from_scheme_funcs[scheme](nside, hp)))

def get_neighbours(scheme, nside, hp):
    return jax.vmap(capture_m1s(partial(to_scheme_funcs[scheme], nside)))(*healpixl_get_neighbours_xy(nside, *from_scheme_funcs[scheme](nside, hp)))

def npix2nside(npix):
    npix = numpy.asarray(npix)
    return numpy.round(numpy.sqrt(npix / 12)).astype(npix.dtype)

def nside2npix(nside):
    return 12*nside*nside

def get_nside(map):
    return npix2nside(len(map))

def convert_map(in_scheme, out_scheme, map):
    nside = get_nside(map)
    permutation = jax.vmap(lambda hp: to_scheme_funcs[in_scheme](nside, *from_scheme_funcs[out_scheme](nside, hp)))(jnp.arange(12*nside*nside))
    return map[permutation]

def ring_above(z, nside):
    """
    Given z = cos(theta) and HEALPix nside,
    returns the ring index just above the input z.
    """
    az = jnp.abs(z)
    # For the equatorial region we simply cast to int.
    equatorial = (nside * (2 - 1.5 * z)).astype(INT_TYPE)
    # For the polar region, compute iring.
    iring = (nside * jnp.sqrt(3 * (1 - az))).astype(INT_TYPE)
    return jnp.where(az <= 2.0 / 3.0, equatorial, jnp.where(z > 0, iring, 4 * nside - iring - 1))

def get_ring_info2(ring, nside):
    """
    For a given ring number and HEALPix nside,
    computes the following:
      - startpix  : starting pixel index for that ring
      - ringpix   : number of pixels in the ring
      - theta     : (characteristic) colatitude of the ring
      - shifted   : boolean flag indicating if the ring uses a half-pixel shift

    Fully branchless: both polar and equatorial quantities are always computed
    and selected with jnp.where. This avoids the jax.lax.cond both-branch
    overhead that occurs when this function is called inside jax.vmap.
    The safety clip and maximum guards are necessary because the branchless
    form evaluates out-of-domain expressions (e.g. arccos for polar rings).
    """
    npix = INT_TYPE(12 * nside * nside)
    ncap = INT_TYPE(2 * nside * (nside - 1))
    fact2 = FLOAT_TYPE(1.0 / (3.0 * nside * nside))
    fact1 = FLOAT_TYPE(2.0 / (3.0 * nside))

    northring = jnp.where(ring > INT_TYPE(2 * nside), INT_TYPE(4 * nside) - ring, ring)
    is_polar = northring < nside

    # Polar cap — always computed; jnp.maximum guards sqrt against negative
    # values that arise when northring is in the equatorial range.
    tmp        = northring * northring * fact2
    costheta_p = FLOAT_TYPE(1.0) - tmp
    sintheta_p = jnp.sqrt(jnp.maximum(tmp * (FLOAT_TYPE(2.0) - tmp), FLOAT_TYPE(0.0)))
    theta_p    = jnp.arctan2(sintheta_p, costheta_p)
    ringpix_p  = INT_TYPE(4) * northring
    startpix_p = INT_TYPE(2) * northring * (northring - INT_TYPE(1))

    # Equatorial — always computed; clip guards arccos against values outside
    # [-1, 1] that arise for polar-cap northring values.
    theta_e    = jnp.arccos(jnp.clip(
                    (INT_TYPE(2 * nside) - northring) * fact1,
                    FLOAT_TYPE(-1.0), FLOAT_TYPE(1.0)))
    ringpix_e  = INT_TYPE(4 * nside)
    shifted_e  = jnp.equal(jnp.mod(northring - nside, INT_TYPE(2)), INT_TYPE(0))
    startpix_e = ncap + (northring - nside) * ringpix_e

    # Select based on region.
    startpix  = jnp.where(is_polar, startpix_p, startpix_e)
    ringpix   = jnp.where(is_polar, ringpix_p,  ringpix_e)
    theta_val = jnp.where(is_polar, theta_p,    theta_e)
    shifted   = jnp.where(is_polar, True,        shifted_e)

    # Adjust for southern hemisphere.
    is_southern = (northring != ring)
    theta_val = jnp.where(is_southern, jnp.pi - theta_val, theta_val)
    startpix  = jnp.where(is_southern, npix - startpix - ringpix, startpix)

    return startpix, ringpix, theta_val, shifted


def _phi_interp(phi, ir, nside, twopi):
    """
    Within a ring, find the two bracketing pixels and the fractional weight.
    Returns (p_left, p_right, w_right, ring_theta).
    w_right is the weight on p_right; w_left = 1 - w_right.
    """
    sp, nr, theta_r, shift = get_ring_info2(ir, nside)
    dphi = twopi / FLOAT_TYPE(nr)
    tmp  = phi / dphi - FLOAT_TYPE(0.5) * shift.astype(FLOAT_TYPE)
    # Equivalent to floor(tmp) but matching the original truncation semantics
    # for negative non-integers (intentionally not jnp.floor).
    i1 = jnp.where(tmp < FLOAT_TYPE(0.0),
                   jnp.trunc(tmp) - FLOAT_TYPE(1.0),
                   jnp.trunc(tmp)).astype(INT_TYPE)
    w  = (phi - (i1.astype(FLOAT_TYPE) + FLOAT_TYPE(0.5) * shift.astype(FLOAT_TYPE)) * dphi) / dphi
    i2 = i1 + INT_TYPE(1)
    i1 = jnp.where(i1 < INT_TYPE(0), i1 + nr, i1)
    i2 = jnp.where(i2 >= nr,         i2 - nr, i2)
    return sp + i1, sp + i2, w, theta_r


def _get_interpol_scalar(theta, phi, nside):
    """
    Compute 4 pixel indices and bilinear interpolation weights for a single
    (theta, phi) pointing.

    Fully branchless: all three cases (general equatorial, north polar cap,
    south polar cap) are computed unconditionally and the correct result is
    selected with jnp.where.  This eliminates the jax.lax.cond both-branch
    overhead that dominates when this function is called inside jax.vmap.

    Safety guards (jnp.where on denominators) are only reached in the two
    polar-cap cases and are discarded by the outer selection in the other case.
    """
    twopi     = FLOAT_TYPE(2.0 * jnp.pi)
    onefourth = FLOAT_TYPE(0.25)

    z   = jnp.cos(theta)
    ir1 = ring_above(z, nside)
    ir2 = ir1 + INT_TYPE(1)

    at_north = (ir1 == INT_TYPE(0))
    at_south = (ir2 == INT_TYPE(4 * nside))

    # Clamp so get_ring_info2 is never called with ring 0 or ring 4*nside.
    ir1_safe = jnp.where(at_north, INT_TYPE(1),                    ir1)
    ir2_safe = jnp.where(at_south, INT_TYPE(4 * nside) - INT_TYPE(1), ir2)

    p0, p1, w1, theta1 = _phi_interp(phi, ir1_safe, nside, twopi)
    p2, p3, w2, theta2 = _phi_interp(phi, ir2_safe, nside, twopi)

    # ------------------------------------------------------------------
    # General equatorial case
    # safe_dt: guards the denominator when the two rings are identical
    # (only happens in a polar-cap pixel; result is discarded by jnp.where).
    # ------------------------------------------------------------------
    safe_dt  = jnp.where(theta2 != theta1, theta2 - theta1, FLOAT_TYPE(1.0))
    wtheta_g = (theta - theta1) / safe_dt
    wg0 = (FLOAT_TYPE(1.0) - wtheta_g) * (FLOAT_TYPE(1.0) - w1)
    wg1 = (FLOAT_TYPE(1.0) - wtheta_g) * w1
    wg2 = wtheta_g * (FLOAT_TYPE(1.0) - w2)
    wg3 = wtheta_g * w2

    # ------------------------------------------------------------------
    # North polar cap (ir1 == 0)
    # Ring ir2_safe == 1: only 4 pixels total.  The "upper" sentinels are
    # the diametrically opposite pixels in that same ring.
    # ------------------------------------------------------------------
    safe_theta2_n = jnp.where(theta2 > FLOAT_TYPE(0.0), theta2, FLOAT_TYPE(1.0))
    wtheta_n = theta / safe_theta2_n
    fac_n    = (FLOAT_TYPE(1.0) - wtheta_n) * onefourth
    wn0 = fac_n
    wn1 = fac_n
    wn2 = (FLOAT_TYPE(1.0) - w2) * wtheta_n + fac_n
    wn3 = w2                     * wtheta_n + fac_n
    pn0 = (p2 + INT_TYPE(2)) & INT_TYPE(3)
    pn1 = (p3 + INT_TYPE(2)) & INT_TYPE(3)
    # pn2, pn3 are p2, p3 (unchanged)

    # ------------------------------------------------------------------
    # South polar cap (ir2 == 4*nside)
    # Ring ir1_safe == 4*nside-1: only 4 pixels total.  The "lower"
    # sentinels are the diametrically opposite pixels in that same ring.
    # ------------------------------------------------------------------
    npix = INT_TYPE(12 * nside * nside)
    safe_denom_s = jnp.where(jnp.pi > theta1, jnp.pi - theta1, FLOAT_TYPE(1.0))
    wtheta_s = (theta - theta1) / safe_denom_s
    ws0 = (FLOAT_TYPE(1.0) - w1) * (FLOAT_TYPE(1.0) - wtheta_s) + wtheta_s * onefourth
    ws1 = w1                     * (FLOAT_TYPE(1.0) - wtheta_s) + wtheta_s * onefourth
    ws2 = wtheta_s * onefourth
    ws3 = wtheta_s * onefourth
    ps2 = ((p0 + INT_TYPE(2)) & INT_TYPE(3)) + (npix - INT_TYPE(4))
    ps3 = ((p1 + INT_TYPE(2)) & INT_TYPE(3)) + (npix - INT_TYPE(4))
    # ps0, ps1 are p0, p1 (unchanged)

    # ------------------------------------------------------------------
    # Select: at_north and at_south are mutually exclusive.
    # p_out[0,1]: north changes sentinel pixels; south keeps p0,p1.
    # p_out[2,3]: north keeps p2,p3; south replaces with south-cap pixels.
    # ------------------------------------------------------------------
    p_out = jnp.stack([
        jnp.where(at_north, pn0, p0),
        jnp.where(at_north, pn1, p1),
        jnp.where(at_north, p2,  jnp.where(at_south, ps2, p2)),
        jnp.where(at_north, p3,  jnp.where(at_south, ps3, p3)),
    ])
    w_out = jnp.stack([
        jnp.where(at_north, wn0, jnp.where(at_south, ws0, wg0)),
        jnp.where(at_north, wn1, jnp.where(at_south, ws1, wg1)),
        jnp.where(at_north, wn2, jnp.where(at_south, ws2, wg2)),
        jnp.where(at_north, wn3, jnp.where(at_south, ws3, wg3)),
    ])
    return p_out, w_out

def _as_float_coords(theta, phi):
    """
    Promote theta/phi to a strong floating dtype.

    Python scalars (and weakly typed traced values) are weakly typed in JAX,
    so arithmetic against the strongly typed FLOAT_TYPE constants in the
    kernels would silently demote a float64 coordinate to float32.  Fixing
    the dtype here keeps float64 inputs float64 and float32 inputs float32.
    """
    dtype = jnp.result_type(theta, phi, float)
    return jnp.asarray(theta, dtype=dtype), jnp.asarray(phi, dtype=dtype)


def get_interpol(theta, phi, nside):
    theta, phi = _as_float_coords(theta, phi)
    # vmap over the scalar interpolation function.
    pix_flat, wgt_flat = jax.vmap(lambda th, ph: _get_interpol_scalar(th, ph, nside), out_axes=1)(theta.ravel(), phi.ravel())
    # Reshape outputs to original input shape with an extra dimension for 4 neighbors.
    new_shape = (4,) + theta.shape
    pix_out = pix_flat.reshape(new_shape)
    wgt_out = wgt_flat.reshape(new_shape)
    return pix_out, wgt_out


# ---------------------------------------------------------------------------
# Precomputed ring table
# ---------------------------------------------------------------------------

def precompute_ring_info(nside):
    """
    Precompute and return ring info for all valid rings (1 … 4*nside-1).

    Returns a tuple (startpix, ringpix, theta, shifted) where each element
    is a JAX array of length 4*nside-1.  The table is 0-indexed: ring r
    corresponds to index r-1.

    Compute this once per nside and pass the result to get_interpol_precomp
    to eliminate transcendental arithmetic (arctan2, arccos, sqrt) from the
    interpolation hot path.

    Example
    -------
    ring_table = precompute_ring_info(nside)
    pix, wgt = jax.jit(get_interpol_precomp)(theta, phi, nside, ring_table)
    """
    rings = jnp.arange(1, INT_TYPE(4 * nside), dtype=INT_TYPE)
    sp, rp, th, sh = jax.vmap(lambda r: get_ring_info2(r, nside))(rings)
    return sp, rp, th, sh


def _phi_interp_table(phi, ir, ring_table, twopi):
    """
    Like _phi_interp but looks up ring info from a precomputed table
    instead of recomputing it.  ir must be >= 1 (caller must clamp).
    """
    sp_tab, rp_tab, th_tab, sh_tab = ring_table
    idx  = ir - INT_TYPE(1)          # table is 0-indexed, rings are 1-indexed
    sp   = sp_tab[idx]
    nr   = rp_tab[idx]
    th_r = th_tab[idx]
    sh   = sh_tab[idx]
    dphi = twopi / FLOAT_TYPE(nr)
    tmp  = phi / dphi - FLOAT_TYPE(0.5) * sh.astype(FLOAT_TYPE)
    i1   = jnp.where(tmp < FLOAT_TYPE(0.0),
                     jnp.trunc(tmp) - FLOAT_TYPE(1.0),
                     jnp.trunc(tmp)).astype(INT_TYPE)
    w    = (phi - (i1.astype(FLOAT_TYPE) + FLOAT_TYPE(0.5) * sh.astype(FLOAT_TYPE)) * dphi) / dphi
    i2   = i1 + INT_TYPE(1)
    i1   = jnp.where(i1 < INT_TYPE(0), i1 + nr, i1)
    i2   = jnp.where(i2 >= nr,         i2 - nr, i2)
    return sp + i1, sp + i2, w, th_r


def _get_interpol_scalar_precomp(theta, phi, nside, ring_table):
    """
    Like _get_interpol_scalar but uses precomputed ring info.
    Eliminates arctan2 / arccos / sqrt from the per-pixel hot path.
    """
    twopi     = FLOAT_TYPE(2.0 * jnp.pi)
    onefourth = FLOAT_TYPE(0.25)

    z   = jnp.cos(theta)
    ir1 = ring_above(z, nside)
    ir2 = ir1 + INT_TYPE(1)

    at_north = (ir1 == INT_TYPE(0))
    at_south = (ir2 == INT_TYPE(4 * nside))

    ir1_safe = jnp.where(at_north, INT_TYPE(1),                    ir1)
    ir2_safe = jnp.where(at_south, INT_TYPE(4 * nside) - INT_TYPE(1), ir2)

    p0, p1, w1, theta1 = _phi_interp_table(phi, ir1_safe, ring_table, twopi)
    p2, p3, w2, theta2 = _phi_interp_table(phi, ir2_safe, ring_table, twopi)

    # General equatorial
    safe_dt  = jnp.where(theta2 != theta1, theta2 - theta1, FLOAT_TYPE(1.0))
    wtheta_g = (theta - theta1) / safe_dt
    wg0 = (FLOAT_TYPE(1.0) - wtheta_g) * (FLOAT_TYPE(1.0) - w1)
    wg1 = (FLOAT_TYPE(1.0) - wtheta_g) * w1
    wg2 = wtheta_g * (FLOAT_TYPE(1.0) - w2)
    wg3 = wtheta_g * w2

    # North polar cap
    safe_theta2_n = jnp.where(theta2 > FLOAT_TYPE(0.0), theta2, FLOAT_TYPE(1.0))
    wtheta_n = theta / safe_theta2_n
    fac_n    = (FLOAT_TYPE(1.0) - wtheta_n) * onefourth
    wn0 = fac_n
    wn1 = fac_n
    wn2 = (FLOAT_TYPE(1.0) - w2) * wtheta_n + fac_n
    wn3 = w2                     * wtheta_n + fac_n
    pn0 = (p2 + INT_TYPE(2)) & INT_TYPE(3)
    pn1 = (p3 + INT_TYPE(2)) & INT_TYPE(3)

    # South polar cap
    npix = INT_TYPE(12 * nside * nside)
    safe_denom_s = jnp.where(jnp.pi > theta1, jnp.pi - theta1, FLOAT_TYPE(1.0))
    wtheta_s = (theta - theta1) / safe_denom_s
    ws0 = (FLOAT_TYPE(1.0) - w1) * (FLOAT_TYPE(1.0) - wtheta_s) + wtheta_s * onefourth
    ws1 = w1                     * (FLOAT_TYPE(1.0) - wtheta_s) + wtheta_s * onefourth
    ws2 = wtheta_s * onefourth
    ws3 = wtheta_s * onefourth
    ps2 = ((p0 + INT_TYPE(2)) & INT_TYPE(3)) + (npix - INT_TYPE(4))
    ps3 = ((p1 + INT_TYPE(2)) & INT_TYPE(3)) + (npix - INT_TYPE(4))

    p_out = jnp.stack([
        jnp.where(at_north, pn0, p0),
        jnp.where(at_north, pn1, p1),
        jnp.where(at_north, p2,  jnp.where(at_south, ps2, p2)),
        jnp.where(at_north, p3,  jnp.where(at_south, ps3, p3)),
    ])
    w_out = jnp.stack([
        jnp.where(at_north, wn0, jnp.where(at_south, ws0, wg0)),
        jnp.where(at_north, wn1, jnp.where(at_south, ws1, wg1)),
        jnp.where(at_north, wn2, jnp.where(at_south, ws2, wg2)),
        jnp.where(at_north, wn3, jnp.where(at_south, ws3, wg3)),
    ])
    return p_out, w_out


def get_interpol_precomp(theta, phi, nside, ring_table):
    """
    Like get_interpol / get_interp_weights but uses a precomputed ring table
    to avoid transcendental arithmetic in the per-pixel hot path.

    Typical usage
    -------------
    ring_table = healjax.precompute_ring_info(nside)   # once per nside

    @jax.jit
    def interp(theta, phi):
        return healjax.get_interpol_precomp(theta, phi, nside, ring_table)

    pix, wgt = interp(theta, phi)
    """
    theta, phi = _as_float_coords(theta, phi)
    pix_flat, wgt_flat = jax.vmap(
        lambda th, ph: _get_interpol_scalar_precomp(th, ph, nside, ring_table),
        out_axes=1,
    )(theta.ravel(), phi.ravel())
    new_shape = (4,) + theta.shape
    return pix_flat.reshape(new_shape), wgt_flat.reshape(new_shape)
