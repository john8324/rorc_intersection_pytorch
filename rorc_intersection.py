# Author: john8324 (https://github.com/john8324)
# Ref 1: https://github.com/opencv/opencv/blob/93d490213fd6520c2af9d2b23f0c99039d355760/modules/imgproc/src/intersection.cpp
# Ref 2: https://github.com/lilanxiao/Rotated_IoU/blob/e2ca1530828ff64c105a53eb23e0788262d72428/box_intersection_2d.py


from typing import Tuple
import torch


def get_rorcs_pts(rorcs: torch.Tensor):
    assert rorcs.shape[1] == 5
    B = rorcs.shape[0]
    DV = rorcs.device

    w, h, a = rorcs[:, 2:].T
    a = torch.deg2rad(a)
    c, s = torch.cos(a), torch.sin(a) # shape = [B]
    R = torch.empty(B, 2, 2, device=DV)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c, s, -s, c
    pts = torch.empty(B, 4, 2, device=DV)
    pts[:, 0, 0], pts[:, 0, 1], pts[:, 1, 0], pts[:, 1, 1] = -w, -h, w, -h
    pts[:, 2, 0], pts[:, 2, 1], pts[:, 3, 0], pts[:, 3, 1] = w, h, -w, h
    pts /= 2
    pts @= R # batch matmul
    pts += rorcs[:, None, :2] # xy
    return pts


def __rotatedRectangleIntersection(rorcs1: torch.Tensor, rorcs2: torch.Tensor, DC_thresh: int=1000000) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    rorcs1: Nx5, rorcs2: Mx5, return NxMx8x2 and NxMx8(torch.bool)
    All dtype = torch.float
    '''
    assert rorcs1.shape[1] == 5
    assert rorcs2.shape[1] == 5
    assert rorcs1.device == rorcs2.device
    assert rorcs1.dtype == rorcs2.dtype
    assert rorcs1.dtype == torch.float
    assert DC_thresh >= 100 # should not be too small

    N, M = rorcs1.shape[0], rorcs2.shape[0]
    #print(N, M)

    # exchange
    if N > M:
        I_pts, mask = __rotatedRectangleIntersection(rorcs2, rorcs1, DC_thresh)
        return I_pts.permute(1, 0, 2, 3), mask.permute(1, 0, 2)
    # zero
    if N == 0:
        return torch.empty(0, M, 8, 2, dtype=torch.float, device=rorcs1.device), torch.empty(0, M, 8, dtype=torch.bool, device=rorcs1.device)
    # too many, devide and conquer
    if N * M > DC_thresh:
        if N == 1:
            hM = M // 2
            I_pts, mask = __rotatedRectangleIntersection(rorcs1, rorcs2[:hM, :], DC_thresh)
            I_pts2, mask2 = __rotatedRectangleIntersection(rorcs1, rorcs2[hM:, :], DC_thresh)
            return torch.hstack([I_pts, I_pts2]), torch.hstack([mask, mask2])
        hN = N // 2
        I_pts, mask = __rotatedRectangleIntersection(rorcs1[:hN, :], rorcs2, DC_thresh)
        I_pts2, mask2 = __rotatedRectangleIntersection(rorcs1[hN:, :], rorcs2, DC_thresh)
        return torch.vstack([I_pts, I_pts2]), torch.vstack([mask, mask2])


    # Main code........................
    pts1, pts2 = get_rorcs_pts(rorcs1), get_rorcs_pts(rorcs2)
    areas1 = (rorcs1[:, 2] * rorcs1[:, 3]).detach_().view(N, 1)
    areas2 = (rorcs2[:, 2] * rorcs2[:, 3]).detach_().view(1, M)

    samePointEps_te = 1e-6 * torch.max(areas1, areas2) # L2 metric

    # Specical case of rect1 == rect2
    flag = torch.abs(pts1[:, None, :, :] - pts2) > samePointEps_te[:, :, None, None] # Nx1x4x2, Mx4x2, NxMx1x1 -> NxMx4x2
    same = ~(flag.view(N, M, 8).any(-1)) #NxM
    same = same.view(N, M, 1).expand(-1, -1, 4) #NxMx4
    same_pts = pts1.view(N, 1, 4, 2) * same.view(N, M, 4, 1).float() #NxMx4x2

    # Line vector
    # A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    vec1 = pts1[:, [1, 2, 3, 0], :] - pts1 # Nx4x2
    vec2 = pts2[:, [1, 2, 3, 0], :] - pts2 # Mx4x2

    #we adapt the epsilon to the smallest dimension of the rects
    nv1 = vec1.detach().norm(2, -1) # Nx4
    nv2 = vec2.detach().norm(2, -1) # Mx4
    minnv1 = torch.min(nv1, 1)[0]
    minnv2 = torch.min(nv2, 1)[0]
    tmp = torch.min(minnv1[:, None], minnv2) #NxM
    samePointEps_te = torch.min(samePointEps_te, tmp)
    tmp[:, :] = 1e-16
    samePointEps_te = torch.max(samePointEps_te, tmp)

    del minnv1, minnv2, tmp, nv1, nv2, areas1, areas2


    # Line test - test all line combos for intersection
    '''
    Find the intersection of line vec1 and vec2
    Ip = p11 + (p12 - p11) * t1 = p21 + (p22 - p21) * t2
    => (p12 - p11) * t1 - (p22 - p21) * t2 = p21 - p11

    Let p12 - p11 = vec1, p22 - p21 = vec2, p21 - p11 = vec3
    => vec1 * t1 - vec2 * t2 = vec3
    =>
    vec1x * t1 - vec2x * t2 = vecx3
    vec1y * t1 - vec2y * t2 = vecy3
    '''
    # Solve for 2x2 Ax=b
    vec3 = pts2[None, :, None, :, :] - pts1[:, None, :, None, :] # NxMx4x4x2

    # det NxMx4x4
    det = vec2[None, :, None, :, 0] * vec1[:, None, :, None, 1] - vec1[:, None, :, None, 0] * vec2[None, :, None, :, 1]
    flag = (~same).view(N, M, 4, 1).repeat(1, 1, 1, 4) # NxMx4x4, no same
    flag = flag & (torch.abs(det) >= 1e-12) # valid det, consider accuracy around 1e-6, i.e. 1e-12 when squared
    t1 = torch.zeros(N, M, 4, 4, dtype=det.dtype, device=det.device)
    t2 = t1.clone()
    t1[flag] = (vec2[None, :, None, :, 0] * vec3[..., 1] - vec3[..., 0] * vec2[None, :, None, :, 1])[flag] / det[flag]
    t2[flag] = (vec1[:, None, :, None, 0] * vec3[..., 1] - vec3[..., 0] * vec1[:, None, :, None, 1])[flag] / det[flag]

    # Intersections exist
    flag = flag & ((t1 >= 0.0) & (t1 <= 1.0)) & ((t2 >= 0.0) & (t2 <= 1.0))
    intersection_pts = (pts1[:, None, :, None, :] + vec1[:, None, :, None, :] * t1[..., None]) * flag[..., None].float()
    intersection_pts = intersection_pts.view(N, M, 16, 2)
    intersection_mask = flag.view(N, M, 16)

    del vec3, det, t1, t2


    # Check for vertices from rect1 inside recct2
    @torch.no_grad()
    def _isOnPositiveSide(line_vec, line_pt, pt):
        '''
        line_pt is root of line_vec
        pt need to be tested

        v = line_pt - pt
        cross(line_vec, -v) = cross(v, line_vec) = v.x * line_vec.y - v.y * line_vec.x
        
        Check cross(v, line_vec) >= 0 => line_vec.y * v.x >= line_vec.x * v.y
        '''
        line_vec, line_pt, pt = torch.broadcast_tensors(line_vec, line_pt, pt)
        v = line_pt - pt
        return line_vec[..., 1] * v[..., 0] >= line_vec[..., 0] * v[..., 1]
    flag = _isOnPositiveSide(vec2[None, :, None, :, :], pts2[None, :, None, :, :], pts1[:, None, :, None, :]) # NxMx4x4
    posSign = flag.count_nonzero(-1) # NxMx4 reduce pts2 dim
    negSign = 4 - posSign

    r1_in_r2_mask = ((posSign == 4) | (negSign == 4)) & ~same # NxMx4
    r1_in_r2_pts = pts1[:, None, :, :] * r1_in_r2_mask[..., None].float() # NxMx4x2

    # Reverse the check - check for vertices from rect2 inside rect1
    flag = _isOnPositiveSide(vec1[:, None, :, None, :], pts1[:, None, :, None, :], pts2[None, :, None, :, :]) # NxMx4x4
    posSign = flag.count_nonzero(-2) # NxMx4 reduce pts1 dim
    negSign = 4 - posSign

    r2_in_r1_mask = ((posSign == 4) | (negSign == 4)) & ~same # NxMx4
    r2_in_r1_pts = pts2[None, ...] * r2_in_r1_mask[..., None].float() # NxMx4x2

    del posSign, negSign, flag
    del pts1, pts2, vec1, vec2

    # merge
    intersection_pts = torch.cat([r1_in_r2_pts, r2_in_r1_pts, intersection_pts], 2) # NxMx24x2
    intersection_mask = torch.cat([r1_in_r2_mask, r2_in_r1_mask, intersection_mask], 2) # NxMx24
    del r1_in_r2_pts, r2_in_r1_pts, r1_in_r2_mask, r2_in_r1_mask

    # Get rid of duplicated points
    K = 24

    with torch.no_grad():
        B = max(min(N * M, DC_thresh // 10), 1)
        # NxMx24x24 is TOO LARGE, therefore use Bx24x24
        # WARNING: the last "B" can be SMALLER than B
        for bi in range(0, N*M, B):
            keep = intersection_mask.view(-1, 24)[bi:bi+B, :].clone() # Bx24

            p = intersection_pts.view(-1, 24, 2)[bi:bi+B, :, :] # Bx24x2
            distPt = torch.square(p[:, :, None, :] - p[:, None, :, :]).sum(-1) # Bx24x24
            del p
            flag = distPt > samePointEps_te.view(-1)[bi:bi+B].view(-1, 1, 1) # Bx24x24
            for k in range(K):
                kj = keep[:, k]
                keep[kj, k+1:] &= flag[kj, k, k+1:]
            del flag, kj

            # set big value for invalid data
            BIG = 1e20
            for k in range(K):
                distPt[:, k, :k+1] = BIG
            tmp = torch.nonzero(~keep)
            distPt[tmp[:, 0], tmp[:, 1], :] = BIG
            distPt[tmp[:, 0], :, tmp[:, 1]] = BIG

            kj = torch.arange(keep.shape[0])
            I_Ns = keep.count_nonzero(-1)
            mm, minI, minJ, minIJ = [None] * 4
            tmp = I_Ns > 8
            while tmp.any(): # we still have duplicate points after samePointEps threshold (eliminate closest points)
                mm, minIJ = torch.min(distPt.view(M, -1), -1)
                minI, minJ = minIJ // K, minIJ % K
                assert (minI != minJ).all() & (mm < BIG).all()

                # drop minJ if n of points > 8
                keep[kj[tmp], minJ[tmp]] = False
                distPt[kj[tmp], minJ[tmp], :] = BIG
                distPt[kj[tmp], :, minJ[tmp]] = BIG
                I_Ns[tmp] -= 1
                tmp = I_Ns > 8

            intersection_mask.view(-1, 24)[bi:bi+B, :] = keep
            assert (intersection_mask.view(-1, 24)[bi:bi+B, :] == keep).all()
            del distPt, kj, mm, minI, minJ, minIJ, tmp, keep


    # compact
    nzmask = intersection_mask.nonzero()
    cnzmask = intersection_mask.count_nonzero(-1)
    intersection_mask2 = torch.arange(K, device=cnzmask.device).view(1, 1, K) < cnzmask.view(N, M, 1)
    nzmask2 = intersection_mask2.nonzero()
    intersection_pts[nzmask2[:, 0], nzmask2[:, 1], nzmask2[:, 2], :] = intersection_pts[nzmask[:, 0], nzmask[:, 1], nzmask[:, 2], :]
    intersection_mask = intersection_mask2
    intersection_pts *= intersection_mask[..., None].float()
    del intersection_mask2, nzmask, nzmask2, cnzmask

    # only 8 points preserved
    I_pts = intersection_pts[:, :, :8, :]
    mask = intersection_mask[:, :, :8]
    del intersection_pts, intersection_mask

    # order points
    idx = argsort_vertice(I_pts, mask)
    # Use no inplace operations
    tmp = torch.gather(I_pts[..., 0], 2, idx), torch.gather(I_pts[..., 1], 2, idx)
    I_pts = torch.stack(tmp, dim=-1)
    mask = torch.gather(mask, 2, idx) # equivalent: mask[i, j, k] = mask[i, j, idx[i, j, k]]
    #for i in range(N):
    #    for j in range(M):
    #        for k in range(8):
    #            I_pts[i, j, k, :] = I_pts[i, j, idx[i, j, k], :]
    #            mask[i, j, k] = mask[i, j, idx[i, j, k]]

    # merge same
    tmp = same[..., 0].nonzero()
    assert ~mask[tmp[:, 0], tmp[:, 1], :].any()
    I_pts[tmp[:, 0], tmp[:, 1], :4, :] = same_pts[tmp[:, 0], tmp[:, 1], :4, :]
    mask[tmp[:, 0], tmp[:, 1], :4] = same[tmp[:, 0], tmp[:, 1], :4]
    return I_pts, mask


@torch.no_grad()
def argsort_vertice(vertices: torch.Tensor, mask: torch.Tensor):
    n = mask.count_nonzero(-1) # NxM
    x = vertices[..., 0].clone() # NxMx8
    y = vertices[..., 1].clone() # NxMx8

    # shift mean to 0
    div = n.max(n.new_ones(n.size())).float() # force n=1 if n=0
    x -= (x.sum(-1) / div).unsqueeze_(-1)
    y -= (y.sum(-1) / div).unsqueeze_(-1)
    
    # sorting
    ang = torch.atan2(y, x) # NxMx8
    # set increasing big value for data not in mask
    ang[~mask] = torch.arange(10, 18, 1, dtype=torch.float, device=ang.device).view(1, 1, 8).repeat(mask.shape[0], mask.shape[1], 1)[~mask]
    index = torch.argsort(ang, dim=-1) # NxMx8
    return index


def rotatedRectangleIntersection(rorcs1: torch.Tensor, rorcs2: torch.Tensor, offset: bool=True, DC_thresh: int=1000000):
    assert rorcs1.dtype == torch.float
    assert rorcs2.dtype == torch.float
    assert rorcs1.shape[1] == 5
    assert rorcs2.shape[1] == 5
    assert rorcs1.device == rorcs2.device
    N, M = rorcs1.shape[0], rorcs2.shape[0]
    empty1, empty2 = (rorcs1[:, 2:4] == 0).all(-1), (rorcs2[:, 2:4] == 0).all(-1)

    # no empty rorcs
    new_rorcs1, new_rorcs2 = rorcs1[~empty1, :], rorcs2[~empty2, :]

    # offset
    average_centers = torch.vstack([rorcs1.detach()[~empty1, :2], rorcs2.detach()[~empty2, :2]]).mean(0) * (1 if offset else 0)
    new_rorcs1[:, :2] -= average_centers ; new_rorcs2[:, :2] -= average_centers

    # Actual run
    I_pts, mask = __rotatedRectangleIntersection(new_rorcs1, new_rorcs2, DC_thresh)

    # restore
    nzm = torch.nonzero(mask)
    I_pts[nzm[:, 0], nzm[:, 1], nzm[:, 2], :] += average_centers
    del nzm

    nzne1, nzne2 = torch.nonzero(~empty1), torch.nonzero(~empty2)
    if nzne1.shape[0] < N or nzne2.shape[0] < M:
        # padding
        nzne = torch.nonzero(~empty1[:, None] & ~empty2)
        New_I_pts = I_pts.new_zeros(N, M, 8, 2)
        New_I_pts[nzne[:, 0], nzne[:, 1], :, :] = I_pts.reshape(-1, 8, 2)
        del I_pts
        New_mask = mask.new_zeros(N, M, 8)
        New_mask[nzne[:, 0], nzne[:, 1], :] = mask.reshape(-1, 8)
        del mask, nzne
        return New_I_pts, New_mask

    return I_pts, mask


def get_intersection_area(intersection: torch.Tensor, mask: torch.Tensor):
    '''
    intersection: NxMx8x2, mask: NxMx8
    '''
    # generate vectors whose roots are their first points
    vec = (intersection - intersection[:, :, 0:1, :]) * mask.float()[..., None] # zero if not mask
    vec1, vec2 = vec[:, :, 1:7, :], vec[:, :, 2:8, :] # sub triangles
    # sum of sub triangle areas
    CROSS = vec1[..., 0] * vec2[..., 1] - vec1[..., 1] * vec2[..., 0]
    return 0.5 * CROSS.abs().sum(-1)



