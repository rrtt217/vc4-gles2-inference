# Compare two engine DUMP files line by line (item 13).
#
# A line is "DUMP <name> w=<texels> h=<hash> [<8 hex chars> ...]"; each hex group
# is one RGBA8 texel holding two little-endian int16 q16 values (XFormat I16_2),
# so the decoder below is the same for a GPU read-back and a CPU x_pack row.
#
# Threshold: |delta| <= 1 LSB is q16 rounding, more is a real difference.  The
# summary prints the first line that exceeds TOL (default 8 LSB).
function hv(c) { return index("0123456789abcdef", c) - 1 }
function val(s, o) { return hv(substr(s,o,1))*16 + hv(substr(s,o+1,1)) }
function s16(a, b,   v) { v = a + 256*b; if (v > 32767) v -= 65536; return v }
NR==FNR {
  nm[FNR] = $2; wd[FNR] = $3; nf[FNR] = NF; hsh[FNR] = $4
  for (i = 5; i <= NF; i++) A[FNR, i] = $i
  n = FNR; next
}
{
  i = FNR
  if (i > n) { printf "line %d: extra line in B (%s)\n", i, $2; next }
  if ($2 != nm[i] || $3 != wd[i]) { printf "line %d: name/width %s %s vs %s %s\n", i, nm[i], wd[i], $2, $3; next }
  if (nf[i] != NF) { printf "line %d %-5s %s: field count %d vs %d (B printed fewer values)\n", i, nm[i], wd[i], nf[i], NF; next }
  if (NF < 5) {
    if (hsh[i] == $4) printf "line %d %-5s %-7s hash equal\n", i, nm[i], wd[i]
    else { printf "line %d %-5s %-7s hash DIFFERS (no bytes)\n", i, nm[i], wd[i]; bad++ }
    next
  }
  maxd = 0; gt1 = 0; eq = 0; cnt = 0; worst = -1; wa = 0; wb = 0
  for (i2 = 5; i2 <= NF; i2++) {
    s = A[i, i2]; t = $i2
    b0 = val(s,1); b1 = val(s,3); b2 = val(s,5); b3 = val(s,7)
    c0 = val(t,1); c1 = val(t,3); c2 = val(t,5); c3 = val(t,7)
    v = s16(b0, b1); u = s16(c0, c1)
    cnt++; if (v == u) eq++; else { d = v - u; if (d < 0) d = -d; if (d > 1) gt1++; if (d > maxd) { maxd = d; worst = (i2-5)*2; wa = v; wb = u } }
    v = s16(b2, b3); u = s16(c2, c3)
    cnt++; if (v == u) eq++; else { d = v - u; if (d < 0) d = -d; if (d > 1) gt1++; if (d > maxd) { maxd = d; worst = (i2-5)*2+1; wa = v; wb = u } }
  }
  flag = (maxd > tol) ? "  <-- beyond rounding" : ""
  printf "line %2d %-5s %-7s n=%d exact=%d |d|>1=%d maxd=%d @%d (%d vs %d)%s\n", i, nm[i], wd[i], cnt, eq, gt1, maxd, worst, wa, wb, flag
  if (maxd > tol && first == 0) { first = i; fn = nm[i]; fd = maxd }
}
END {
  if (first) printf "\nfirst beyond %d LSB: line %d (%s) maxd=%d\n", tol, first, fn, fd
  else printf "\nno line exceeds %d LSB\n", tol
}
